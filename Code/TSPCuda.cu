#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "TSPCuda.h"
//#include "timer.h"
#include "globalData.h"
//#include "IndMPINode.h"

#define CHECK_GPU(msg) check_gpu__ (__FILE__, __LINE__, (msg))
static void check_gpu__ (const char * file, size_t line, const char * msg);
static void check_gpu__ (const char* file, size_t line, const char* msg)
{
  cudaError_t err = cudaGetLastError ();
  if (err != cudaSuccess) {
    fprintf (stderr, "*** [%s:%lu] %s -- CUDA Error (%d): %s ***\n",
	     file, line, msg, (int)err, cudaGetErrorString (err));
    exit (-1);
  }
}

__global__
void TSPSwapKernel (unsigned int n, int* A, int* coords, unsigned int loops)
{
 
  int currentNumCities=0;
  currentNumCities = NUM_CITIES / 4;

  if(blockIdx.x < (NUM_CITIES % 4))
	currentNumCities++;

   __shared__ float fitnessMatrix[(NUM_CITIES/4)+1][(NUM_CITIES/4)+1];
  int localTour[currentNumCities];
  int i, prevX, prevY, city;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int bid = blockIdx.x;
  float distance = 0;
  int temp;

  for(i = 0; i < currentNumCities; i++)
    localTour[i] = (bid * blockDim.x) + i;
  
  if(tidx <= tidy && tidx != 0 && tidy != blockDim.x)
  {
    //swap cities tidx and tidy
    temp = localTour[tidx];
    localTour[tidx] = localTour[tidy];
    localTour[tidy] = temp;
    
    prevX = (coords + (3 * localTour[0]))[1];
    prevY = (coords + (3 * localTour[0]))[2];

    for(i = 1; i < currentNumCities; i++)
    {
      city = localTour[i];
      distance += (float)(((coords + (city * 3))[2] - prevY) * ((coords + (city * 3))[2] - prevY)) 
	           + (float)(((coords + (3 * city))[1] - prevX) * ((coords + (3 * city))[1] - prevX));
      prevX = (coords + (city * 3))[1];
      prevY = (coords + (city * 3))[2];
    }

    fitnessMatrix[tidx][tidy] = distance;
  }    
}


int *
createArrayOnGPU (unsigned int n)
{
  int* tour_gpu = NULL;
  if (n) {
    cudaMalloc (&tour_gpu, n * sizeof (int));
    CHECK_GPU ("Out of memory?");
    assert (tour_gpu);
  }
  return tour_gpu;
}

int* createDMatOnGPU(unsigned int n)
{
  int* coords_dMat = NULL;
  if(n)
  {
    cudaMalloc(&coords_dMat,3 * n * sizeof(int));
    CHECK_GPU("OUT OF MEMORY FOR DMAT");
    assert(coords_dMat);
  }
  return coords_dMat;
}

void
freeKeysOnGPU (int* tour_gpu)
{
  if (tour_gpu) cudaFree (tour_gpu);
}

void
copyKeysToGPU (unsigned int n, int* Dest_gpu, const int* Src_cpu)
{
  cudaMemcpy (Dest_gpu, Src_cpu, n * sizeof (int),
              cudaMemcpyHostToDevice);  CHECK_GPU ("Copying keys to GPU");
}

void copyDMatToGPU(unsigned int n, int * Dest_gpu, const int** Src_cpu)
{
  int i;
  for(i = 0; i < n; i++)
  {
    cudaMemcpy(Dest_gpu + (i * 3), Src_cpu[i], 3 * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_GPU ("Copying coords to GPU");
  }
}

void
copyKeysFromGPU (unsigned int n, int* Dest_cpu, const int* Src_gpu)
{
  cudaMemcpy (Dest_cpu, Src_gpu, n * sizeof (int),cudaMemcpyDeviceToHost);  
  CHECK_GPU ("Copying keys from GPU");
}

extern "C"
void TSPSwapRun(int* tour,const int** coords)
{
  int n = NUM_CITIES;
  int* tour_gpu = createArrayOnGPU(n);
  int* coords_gpu = createDMatOnGPU(NUM_CITIES);

  copyKeysToGPU (n,tour_gpu, tour);
  copyDMatToGPU(n, coords_gpu, coords);

  /* Start timer, _after_ CPU-GPU copies */
 // stopwatch_t* timer = stopwatch_create ();
 // assert (timer);
 // stopwatch_start (timer);

  TSPSwapKernel<<<NUM_BLOCKS_CUDA,BLOCKSIZE_CUDA>>> (n, tour_gpu, coords_gpu, localIter);
  cudaDeviceSynchronize();

  /* Stop timer and report bandwidth _without_ the CPU-GPU copies */
 //long double t_merge_nocopy = stopwatch_stop (timer);
}

/*int main()
{
  TSPSwapRun((int *)0, (const int**)0);
}*/
