#include "globalData.h"
#define MAX_ITERATIONS 20
#define NUM_THREADS 32
const unsigned int NUM_BLOCKS_CUDA = 4;
const unsigned int BLOCKSIZE_CUDA = NUM_CITIES*NUM_CITIES;

int* createKeysOnGPU (unsigned int n);

/** Free memory previously allocated for keys on the GPU. */
void freeKeysOnGPU (int* A);

/** Transfers keys to the GPU, i.e., a wrapper around cudaMemcpy. */
void copyKeysToGPU (unsigned int n,
                       int* Dest_gpu, const int* Src_cpu);

/** Copy keys from the GPU, i.e., a wrapper around cudaMemcpy. */
void copyKeysFromGPU (unsigned int n,
                      int* Dest_cpu, const int* Src_gpu);

