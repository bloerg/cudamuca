
// OpenCL
#include <CL/cl.hpp>

// This includes my_uint64 type
#include "ising2D_io.hpp"

#include "muca.hpp"

// Random Number Generator
#include "Random123/philox.h"
#include "Random123/examples/uniform.hpp"

// 256 threads per block ensures the possibility of full occupancy
// for all compute capabilities if thread count small enough
#define WORKERS_PER_BLOCK 256

// choose random number generator
typedef r123::Philox4x32_R<7> RNG;

using namespace std;

int main(int argc, char** argv) 
{
  parseArgs(argc, argv);
  if (NUM_WORKERS % WORKERS_PER_BLOCK != 0) {
    cerr << "ERROR: NUM_WORKERS must be multiple of " << WORKERS_PER_BLOCK << endl;
  }


  // select device
  vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
      cout << "No platforms found. Check OpenCL installation!\n";
      exit(1);
  }
  cl::Platform default_platform=all_platforms[0]; //1
  cout << "\nUsing platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

  vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices); 
  int deviceCount = all_devices.size();
  if (deviceCount == 0) {
      cout << "No devices found. Check OpenCL installation!\n";
      exit(1);
  }

  cl::Device device;
  if(REQUESTED_GPU >= 0 and REQUESTED_GPU < deviceCount)
  {
    device = all_devices[REQUESTED_GPU];
  }
  else 
  {
    device = all_devices[0];
  }
  cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";


  //~ // prefer cache over shared memory
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  // figure out optimal execution configuration
  // based on GPU architecture and generation
  int maxresidentthreads = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  int totalmultiprocessors = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  int optimum_number_of_workers = maxresidentthreads*totalmultiprocessors;
  if (NUM_WORKERS == 0) {
    NUM_WORKERS = optimum_number_of_workers;
  }
  cout << "GPU capabilities\nCL_DEVICE_MAX_WORK_GROUP_SIZE: " << maxresidentthreads << "\nCL_DEVICE_MAX_COMPUTE_UNITS: " << totalmultiprocessors << "\n";
 
  cl::Context cl_context({device});
  cl::CommandQueue cl_queue(cl_context, device);

 // read the kernel from source file
  std::ifstream cl_program_file_ising("ising2D_cl.cl");
  std::string cl_program_string_ising(
      std::istreambuf_iterator<char>(cl_program_file_ising),
      (std::istreambuf_iterator<char>())
  );

 
  cl::Program cl_program_ising(cl_context, cl_program_string_ising, true);
  
  if (cl_program_ising.build({ device }, "-I include") != CL_SUCCESS){
      std::cout << " Error building: " << cl_program_ising.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
      getchar();
      exit(1);
  }
  
  cl::Kernel cl_kernel_compute_energies(cl_program_ising, "computeEnergies");
  cl::Kernel cl_kernel_muca_iteration(cl_program_ising, "mucaIteration");



  //~ cout << "N: "<< N << " L: " << L << " NUM_WORKERS: " << NUM_WORKERS << "\n"; //debug
  
  
  // initialize NUM_WORKERS (LxL) lattices
  RNG rng;
  vector<cl_char> h_lattice(NUM_WORKERS * N);
  //~ cl_char* d_lattice;
  //~ cudaMalloc((void**)&d_lattice, NUM_WORKERS * N * sizeof(int8_t));
  for (unsigned worker=0; worker < NUM_WORKERS; worker++) {
    RNG::key_type k = {{worker, 0xdecafbad}};
    RNG::ctr_type c = {{0, seed, 0xBADCAB1E, 0xBADC0DED}};
    RNG::ctr_type r;
    for (size_t i = 0; i < N; i++) {
      if (i%4 == 0) {
        ++c[0];
        r = rng(c, k);
      }
      h_lattice.at(i*NUM_WORKERS+worker) = 2*(r123::u01fixedpt<float>(r.v[i%4]) < 0.5)-1;
    }
  }
  //~ cudaMemcpy(d_lattice, h_lattice.data(), NUM_WORKERS * N * sizeof(int8_t), cudaMemcpyHostToDevice);
  
  int memory_operation_status;
  cl::Event writeEvt;
  
  cl::Buffer d_lattice_buf (
    cl_context,
    CL_MEM_READ_WRITE,
    NUM_WORKERS * N * sizeof(cl_char),
    NULL,
    &memory_operation_status
  );
  std::cout << "clCreateBuffer status: " << memory_operation_status << "\n";
  memory_operation_status = cl_queue.enqueueWriteBuffer(d_lattice_buf, CL_TRUE, 0, NUM_WORKERS * N * sizeof(cl_char), h_lattice.data(), NULL, &writeEvt);
  std::cout << "clEnqueueWriteBuffer status: " << memory_operation_status << "\n";


  // initialize all energies
  //~ int* d_energies;
  //~ cudaMalloc((void**)&d_energies, NUM_WORKERS * sizeof(int));

  cl::Buffer d_energies_buf (
    cl_context,
    CL_MEM_READ_WRITE,
    NUM_WORKERS * sizeof(cl_int),
    NULL,
    &memory_operation_status
  );
  std::cout << "clCreateBuffer status: " << memory_operation_status << "\n";
  
  cl_kernel_compute_energies.setArg(0, d_lattice_buf);
  cl_kernel_compute_energies.setArg(1, d_energies_buf);
  cl_kernel_compute_energies.setArg(2, &L);
  cl_kernel_compute_energies.setArg(3, &N);
  cl_kernel_compute_energies.setArg(4, &NUM_WORKERS);

  //FIXME: Machen diese Parameter Sinn?
  cl_queue.enqueueNDRangeKernel(cl_kernel_compute_energies, cl::NDRange(0), cl::NDRange(NUM_WORKERS / WORKERS_PER_BLOCK), cl::NDRange(WORKERS_PER_BLOCK));


  // initialize ONE global weight array
  //~ vector<cl_float> h_log_weights(N + 1, 0.0f); //gives a warning at compile time
  vector<float> h_log_weights(N + 1, 0.0f); // no warning
  //~ cudaMalloc((void**)&d_log_weights, (N + 1) * sizeof(float));
  cl::Buffer d_log_weights_buf (
    cl_context,
    CL_MEM_READ_WRITE,
    ( N + 1 ) * sizeof(cl_float),
    NULL,
    &memory_operation_status
  );
  std::cout << "clCreateBuffer status: " << memory_operation_status << "\n";
  
  
 
  
  // initialize ONE global histogram
  vector<my_uint64> h_histogram((N + 1), 99); //FIXME: must by 0 instead of 99
  //~ my_uint64* d_histogram;
  //~ cudaMalloc((void**)&d_histogram, (N + 1) * sizeof(my_uint64));
  cl::Buffer d_histogram_buf (
    cl_context,
    CL_MEM_READ_WRITE,
    ( N + 1 ) * sizeof(my_uint64),
    NULL,
    &memory_operation_status
  );
  std::cout << "clCreateBuffer status: " << memory_operation_status << "\n";
  memory_operation_status = cl_queue.enqueueWriteBuffer(d_histogram_buf, CL_TRUE, 0, (N+1) * sizeof(my_uint64), h_histogram.data(), NULL, &writeEvt);
  std::cout << "clEnqueueWriteBuffer status: " << memory_operation_status << "\n";
  
  
  // timing and statistics
  vector<long double> times;
  timespec start, stop;
  ofstream iterfile;
  
  iterfile.open("run_iterations.dat");
  // initial estimate of width at infinite temperature 
  // (random initialization requires practically no thermalization)
  unsigned width = 10;
  double nupdates_run = 1;
  // heuristic factor that determines the number of statistic per iteration
  // should be related to the integrated autocorrelation time
  double z = 2.25;
  
    
  // main iteration loop
  cl::Event readEvt;
  for (cl_uint k=0; k < MAX_ITER; k++) {
    cout << "max_iter: " << MAX_ITER << " k: " << k << "\n";
    // start timer
    //~ cudaDeviceSynchronize();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    // copy global weights to GPU
    //~ cudaMemcpy(d_log_weights, h_log_weights.data(), (N + 1) * sizeof(float), cudaMemcpyHostToDevice);

    memory_operation_status = cl_queue.enqueueWriteBuffer(d_log_weights_buf, CL_TRUE, 0, (N+1) * sizeof(cl_float), h_log_weights.data(), NULL, &writeEvt);
    std::cout << "clEnqueueWriteBuffer status: " << memory_operation_status << "\n";
    
    
    // acceptance rate and correlation time corrected "random walk"
    // in factor 30 we adjusted acceptance rate and >L range requirement of our present Ising situation
    NUPDATES_THERM = 30 * width;
    if (width<N) {
      // 6 is motivated by the average acceptance rate of a multicanonical simulation ~0.45 -> (1/0.45)**z~6
      nupdates_run = 6 * pow(width, z) / NUM_WORKERS;
    }
    else {
      // for a flat spanning histogram, we assume roughly equally distributed
      // walkers and reduce the thermalization time
      // heuristic modification factor;
      // (>1; small enough to introduce statistical fluctuations on the convergence measure)
      nupdates_run *= 1.1;
    }
    NUPDATES = static_cast<my_uint64>(nupdates_run)+1;
    // local iteration on each thread, writing to global histogram
    
    //FIXME: //~ mucaIteration<<<NUM_WORKERS / WORKERS_PER_BLOCK, WORKERS_PER_BLOCK>>>(d_lattice, d_histogram, d_energies, k, seed, NUPDATES_THERM, NUPDATES);
    
    cl_kernel_muca_iteration.setArg(0, d_lattice_buf);
    cl_kernel_muca_iteration.setArg(1, d_histogram_buf);
    cl_kernel_muca_iteration.setArg(2, d_energies_buf);
    cl_kernel_muca_iteration.setArg(3, d_log_weights_buf);
    cl_kernel_muca_iteration.setArg(4, &k);
    cl_kernel_muca_iteration.setArg(5, &seed);
    cl_kernel_muca_iteration.setArg(6, &NUPDATES_THERM);
    cl_kernel_muca_iteration.setArg(7, &NUPDATES);
    cl_kernel_muca_iteration.setArg(8, &L);
    cl_kernel_muca_iteration.setArg(9, &N);
    cl_kernel_muca_iteration.setArg(10, &NUM_WORKERS);
    
    //FIXME: Machen diese Parameter Sinn?
    //~ cl_queue.enqueueNDRangeKernel(cl_kernel_muca_iteration, cl::NDRange(0), cl::NDRange(NUM_WORKERS / WORKERS_PER_BLOCK), cl::NDRange(WORKERS_PER_BLOCK));
    cl_queue.enqueueNDRangeKernel(cl_kernel_muca_iteration, 1, cl::NDRange(1), cl::NDRange(1));

    cout << "iteration: " << k << "\n";
    //~ cudaError_t err = cudaGetLastError();
    //~ if (err != cudaSuccess) {
      //~ cout << "Error: " << cudaGetErrorString(err) << " in " << __FILE__ << __LINE__ << endl;
      //~ exit(err);
    //~ }

    // copy global histogram back to CPU
    //~ cudaMemcpy(h_histogram.data(), d_histogram, (N + 1) * sizeof(my_uint64), cudaMemcpyDeviceToHost);
    
    //~ memory_operation_status = cl_queue.enqueueReadBuffer(d_histogram_buf, CL_TRUE, 0, ( N + 1 ) * sizeof(my_uint64), h_histogram.data(), NULL, &readEvt);
    memory_operation_status = cl_queue.enqueueReadBuffer(d_histogram_buf, CL_TRUE, 0, ( N + 1 ) * sizeof(my_uint64), h_histogram.data());
    
    cout << "readstatus: " << memory_operation_status << "\n";
    
    for (uint hc = 0 ; hc < N+1; hc++) {
      std::cout << "hc: " << hc << ", hist[" << hc << "]: " << h_histogram.at(hc) << "\n";
    }
    
    // stop timer
    //~ cudaDeviceSynchronize();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    long double elapsed = 1e9* (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);
    times.push_back(elapsed);
    TOTAL_THERM   += NUPDATES_THERM;
    TOTAL_UPDATES += NUPDATES;
   
    // flatness in terms of kullback-leibler-divergence; 
    // requires sufficient thermalization!!! 
    double dk  = d_kullback(h_histogram);
    iterfile << "#NITER = " << k  << " dk=" << dk << endl;
    writeHistograms(h_log_weights, h_histogram, iterfile);
    if (dk<1e-4) {
      break;
    }

    // measure width of the current histogram
    size_t start,end;
    getHistogramRange(h_histogram, start, end);
    unsigned width_new = end-start;
    if (width_new > width) width=width_new;
    // update logarithmic weights with basic scheme if not converged
    updateWeights(h_log_weights, h_histogram);
  }
  iterfile.close();
  ofstream sout;
  sout.open("stats.dat");
  writeStatistics(times, sout);
  sout << "total number of thermalization steps/Worker : " << TOTAL_THERM << "\n";
  sout << "total number of iteration updates   /Worker : " << TOTAL_UPDATES << "\n";
  sout << "total number of all updates         /Worker : " << TOTAL_THERM+TOTAL_UPDATES << "\n";
  sout.close();
  


//free memory section

 

return 0;
}
