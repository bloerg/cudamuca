
// OpenCL
#include <CL/cl.hpp>

// This includes my_uint64 type
#include "ising2D_io.hpp"

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




//free memory section

return 0;
}
