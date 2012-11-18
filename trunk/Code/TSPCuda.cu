#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "TSPCuda.h"
#include "timer.h"
#include "globalData.h"

__global__
void TSPSwapKernel (unsigned int n, keytype* A, int* coords, unsigned int loops)
{
  extern __shared__ float fitnessMatrix[NUM_CITIES/4][NUM_CITIES/4];
  int localTour[NUM_CITIES/4];
  int i, prevX, prevY, city;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int bid = blockIdx.x;
  float distance = 0;
  int temp;

  for(i = 0; i < NUM_CITIES/4; i++)
    localTour[i] = (bid * blockDim.x) + i;
  
  if(tidx <= tidy && tidx != 0 && tidy != blockDim.x)
  {
    //swap cities tidx and tidy
    temp = localTour[tidx];
    localTour[tidx] = localTour[tidy];
    localTour[tidy] = temp;
    
    prevX = (coords + (3 * localTour[0]))[1];
    prevY = (coords + (3 * localTour[0]))[2];

    for(i = 1; i < NUM_CITIES/4; i++)
    {
      city = localTour[i];
      distance += (float)pow(((coords + (city * 3))[2] - prevY),2) + (float)pow(((coords + (3 * city))[1] - prevX),2);
      prevX = (coords + (city * 3))[1];
      prevY = (coords + (city * 3))[2];
    }

    fitnessMatrix[tidx][tidy] = distance;
  }    
}


ketype *
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
    assert(tour_dMat);
  }
  return tour_dMat;
}

void
freeKeysOnGPU (keytype* tour_gpu)
{
  if (tour_gpu) cudaFree (tour_gpu);
}

void
copyKeysToGPU (unsigned int n, keytype* Dest_gpu, const keytype* Src_cpu)
{
  cudaMemcpy (Dest_gpu, Src_cpu, n * sizeof (keytype),
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
  cudaMemcpy (Dest_cpu, Src_gpu, n * sizeof (keytype),cudaMemcpyDeviceToHost);  
  CHECK_GPU ("Copying keys from GPU");
}

void TSPSwapRun(int* tour, int** coords)
{
  int n = NUM_CITIES;
  int* tour_gpu = createArrayOnGPU(n);
  int* coords_gpu = createDMatOnGPU(NUM_CITIES);

  copyKeysToGPU (n,tour_gpu, tour);
  copyDMatToGPU(n, tour_dMat, coords);

  /* Start timer, _after_ CPU-GPU copies */
  stopwatch_t* timer = stopwatch_create ();
  assert (timer);
  stopwatch_start (timer);

  TSPSwapKernel<<<NUM_BLOCKS, BLOCKSIZE>>> (n, tour_gpu, coords_gpu, localIter);
  cudaDeviceSynchronize();

  /* Stop timer and report bandwidth _without_ the CPU-GPU copies */
  long double t_merge_nocopy = stopwatch_stop (timer);
}

