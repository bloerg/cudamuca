# Makefile for compilation of gcc and cuda version of pmuca ising2D

# please select your mpi compiler
MPICC=mpic++
CPU_FLAGS=-pedantic -Wall -Wextra -O3 -std=c++0x -I./Random123/include/

# please set this to your cuda path
ifeq ($(wildcard /net/nfs/opt/cuda/bin/nvcc),) 
  NVCC=nvcc
else
  NVCC=/net/nfs/opt/cuda/bin/nvcc
endif
GPU_ARCHS=-arch=sm_35 -rdc=true -I./Random123/include/ -lineinfo
GPU_FLAGS=-Xcompiler -Wall,-Wno-unused-function,-O3

# opencl path
export CPLUS_INCLUDE_PATH=/net/nfs/opt/opencl/1.2/include
export LIBRARY_PATH=/net/nfs/opt/opencl/1.2/include

all: gpu cpu

gpu: ising2D_gpu

cpu: ising2D_cpu

ising2D_gpu: ising2D.cu
	$(NVCC) $(GPU_ARCHS) $(GPU_FLAGS) --ptxas-options=-v  ising2D.cu -o $@

ising2D_cpu: ising2D.cpp
	$(MPICC) $(CPU_FLAGS) ising2D.cpp -o $@

ising2D_cl: ising2D_cl.cpp
	g++ -lOpenCL  -std=gnu++0x $(OPT) ising2D_cl.cpp  -o $@
	

clean:
	rm -f ising2D_gpu
	rm -f ising2D_cpu
	rm -f ising2D_cl
