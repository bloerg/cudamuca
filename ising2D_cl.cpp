// This includes my_uint64 type
#include "ising2D_io.hpp"

// 256 threads per block ensures the possibility of full occupancy
// for all compute capabilities if thread count small enough
#define WORKERS_PER_BLOCK 256;


using namespace std;

int main(int argc, char** argv) 
{
  parseArgs(argc, argv);
  if (NUM_WORKERS % WORKERS_PER_BLOCK != 0) {
    cerr << "ERROR: NUM_WORKERS must be multiple of " << WORKERS_PER_BLOCK << endl;
  }

//free memory section

}
