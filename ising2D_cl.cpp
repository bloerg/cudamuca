
// OpenCL
#include <CL/cl.hpp>

// This includes my_uint64 type
#include "ising2D_io.hpp"

// 256 threads per block ensures the possibility of full occupancy
// for all compute capabilities if thread count small enough
#define WORKERS_PER_BLOCK 256


using namespace std;

int main(int argc, char** argv) 
{
  parseArgs(argc, argv);
  if (NUM_WORKERS % WORKERS_PER_BLOCK != 0) {
    cerr << "ERROR: NUM_WORKERS must be multiple of " << WORKERS_PER_BLOCK << endl;
  }


  // select device
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
      std::cout << "No platforms found. Check OpenCL installation!\n";
      exit(1);
  }
  cl::Platform default_platform=all_platforms[1];
  std::cout << "\nUsing platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices); 
  int deviceCount = all_devices.size();
  if (deviceCount == 0) {
      std::cout << "No devices found. Check OpenCL installation!\n";
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
  std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";


  //~ // prefer cache over shared memory
  //~ cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  // figure out optimal execution configuration
  // based on GPU architecture and generation
  int maxresidentthreads = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  int totalmultiprocessors = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  int optimum_number_of_workers = maxresidentthreads*totalmultiprocessors;
  if (NUM_WORKERS == 0) {
    NUM_WORKERS = optimum_number_of_workers;
  }

  std::cout << "GPU capabilities\nCL_DEVICE_MAX_WORK_GROUP_SIZE: " << maxresidentthreads << "\nCL_DEVICE_MAX_COMPUTE_UNITS: " << totalmultiprocessors << "\n";
  
  cl::Context cl_context({device});




//free memory section

return 0;
}
