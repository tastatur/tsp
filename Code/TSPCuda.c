#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "TSPCuda.h"
#include "timer.h"


__global__
void TSPSwap (unsigned int n, keytype** A, unsigned int offset)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float temp = A[gid][gid];
  A[gid][gid] = A[gid+offset][gid+offset];
  A[gid+offset][gid+offset] = temp;
  
}


ketype **
createArrayOnGPU (unsigned int n)
{
  float** A_gpu = NULL;
  if (n) {
    cudaMalloc (&A_gpu, n * sizeof (keytype**)); CHECK_GPU ("Out of memory?");
    assert (A_gpu);
  }
  return A_gpu;
}

void
freeKeysOnGPU (keytype** A_gpu)
{
  if (A_gpu) cudaFree (A_gpu);
}

void
copyKeysToGPU (unsigned int n, keytype* Dest_gpu, const keytype* Src_cpu)
{
  cudaMemcpy (Dest_gpu, Src_cpu, n * sizeof (keytype),
              cudaMemcpyHostToDevice);  CHECK_GPU ("Copying keys to GPU");
}

void
copyKeysFromGPU (unsigned int n, keytype* Dest_cpu, const keytype* Src_gpu)
{
  cudaMemcpy (Dest_cpu, Src_gpu, n * sizeof (keytype),
              cudaMemcpyDeviceToHost);  CHECK_GPU ("Copying keys from GPU");
}

void TSPSwapRun()
{

keytype* A_gpu = createKeysOnGPU (n);
  copyKeysToGPU (n, A_gpu, A);

  /* Start timer, _after_ CPU-GPU copies */
  stopwatch_t* timer = stopwatch_create (); assert (timer);
  stopwatch_start (timer);

//  const unsigned int n_half = n >> 1; /* n/2 */
  const unsigned int BLOCKSIZE = (NUM_CITIES / 4*2) ;  /* threads per block */
  const unsigned int NUM_BLOCKS = 4;
  assert (isPower2 (n) && isPower2 (BLOCKSIZE));

  //unsigned int offset = n_half;
  //while (offset >= BLOCKSIZE){
        TSP<<<NUM_BLOCKS, BLOCKSIZE>>> (n, A_gpu, offset);
    //    offset >>= 1;
  //}

  cudaDeviceSynchronize();

  /* Stop timer and report bandwidth _without_ the CPU-GPU copies */
  long double t_merge_nocopy = stopwatch_stop (timer);


}

