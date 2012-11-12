keytype* createKeysOnGPU (unsigned int n);

/** Free memory previously allocated for keys on the GPU. */
void freeKeysOnGPU (keytype* A);

/** Transfers keys to the GPU, i.e., a wrapper around cudaMemcpy. */
void copyKeysToGPU (unsigned int n,
                       keytype* Dest_gpu, const keytype* Src_cpu);

/** Copy keys from the GPU, i.e., a wrapper around cudaMemcpy. */
void copyKeysFromGPU (unsigned int n,
                      keytype* Dest_cpu, const keytype* Src_gpu);

