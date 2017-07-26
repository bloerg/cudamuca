
// 256 threads per block ensures the possibility of full occupancy
// for all compute capabilities if thread count small enough
#define WORKERS_PER_BLOCK 256
#define WORKER get_global_id(0)


// from ising2D_io.hpp
typedef unsigned long long my_uint64;
//~ __constant double my_uint64_max = 18446744073709551615; // = pow(2.0,64)-1;


// calculate energy difference of one spin flip
inline int localE(uint idx, __global char* lattice, int* d_L, int* d_N, int* d_NUM_WORKERS)
{ 
  int right = idx + 1;
  int left = convert_int( idx ) -1; 

  int up = idx + *d_L;
  int down = convert_int( idx ) - *d_L;
  
  // check periodic boundary conditions
  if (right % *d_L == 0) right -= *d_L;
  if (idx % *d_L == 0) left += *d_L;
  if (up > convert_int(*d_N - 1) ) up -= *d_N;
  if (down < 0 ) down += *d_N;
   
  return -lattice[idx * *d_NUM_WORKERS + WORKER] *
     ( lattice[right * *d_NUM_WORKERS + WORKER] +
       lattice[left * *d_NUM_WORKERS + WORKER] +
       lattice[up * *d_NUM_WORKERS + WORKER] + 
       lattice[down * *d_NUM_WORKERS + WORKER] );
}

// calculate total energy
int calculateEnergy(__global char* lattice, int* d_L, int* d_N, int* d_NUM_WORKERS)
{
  int sum = 0;

  for (size_t i = 0; i < *d_N; i++) {
    sum += localE(i, lattice, d_L, d_N, d_NUM_WORKERS);
  }
  // divide out double counting
  return (sum >> 1); 
}

__kernel void computeEnergies(__global char* d_lattice, __global int* d_energies, __private int d_L, __private int d_N, __private int d_NUM_WORKERS)
{
  d_energies[WORKER] = calculateEnergy(d_lattice, &d_L, &d_N, &d_NUM_WORKERS);
}
