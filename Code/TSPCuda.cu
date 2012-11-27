#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "TSPCuda.h"
#include "globalData.h"

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
void TSPSwapKernel (unsigned int n, int* completeTour, int* coords, unsigned int loops)
{
 
  int currentNumCities = 0;
  currentNumCities = NUM_CITIES / 4;

   __shared__ float fitnessMatrix[(NUM_CITIES/4) + 1][(NUM_CITIES/4) + 1];
   __shared__ int globalTourThreads[(NUM_CITIES/4) + 1];
   __shared__ int swapCities[2] ;

  int localTour[(NUM_CITIES/4)+1];
  int i,j, prevX, prevY, city , Min = 0 ,  counter = 0;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int bid = blockIdx.x;
  float distance = 0;
  float distanceBackup = 0;
  float improvement = 0;
  int  offset = 0;
  int rem = NUM_CITIES % 4;
  int temp;
 

  offset = (NUM_CITIES/4) * bid;
  if(bid < rem)
  {
    currentNumCities++;
    offset += bid;
  }
  if(bid >= rem)
    offset += rem;
  
  for(i = 0; i < currentNumCities; i++) {
    localTour[i] = completeTour[offset + i];
    globalTourThreads[i] = localTour[i];
  }

  prevX = (coords + (2 * localTour[0]))[0];
  prevY = (coords + (2 * localTour[0]))[1];
  
  for(i = 1; i < currentNumCities; i++)
  {
    city = localTour[i];
    distanceBackup += (float)(((coords + (city * 2))[1] - prevY) * ((coords + (city * 2))[1] - prevY))
      + (float)(((coords + (2 * city))[0] - prevX) * ((coords + (2 * city))[0] - prevX));
    prevX = (coords + (city * 2))[0];
    prevY = (coords + (city * 2))[1];
  }  
  fitnessMatrix[tidx][tidy] = (float)INT_MAX;


  while ( counter != MAX_ITERATIONS )
  {  
    improvement = 0;
    swapCities[0] = 0;
    swapCities[1] = 0;

    if(tidx < tidy && tidx != 0 && tidy != currentNumCities)
    {
      /* 
       * what is in globalTourThreads for the 
       * first iteration ?
       */
      for(i = 0; i < currentNumCities; i++) {
      	localTour[i] = globalTourThreads[i];
      }

      //swap cities tidx and tidy
      temp = localTour[tidx];
      localTour[tidx] = localTour[tidy];
      localTour[tidy] = temp;
      
      prevX = (coords + (2 * localTour[0]))[0];
      prevY = (coords + (2 * localTour[0]))[1];
      
      distance = 0;
      
      for(i = 1; i < currentNumCities; i++)
      {
      	int city = localTour[i];
      	distance += (((coords + (city * 2))[1] - prevY) * ((coords + (city * 2))[1] - prevY))
	+ (float)(((coords + (2 * city))[0] - prevX) * ((coords + (2 * city))[0] - prevX));

      	prevX = (coords + (city * 2))[0];
      	prevY = (coords + (city * 2))[1];
      }
      
      //fitnessMatrix[tidx][tidy] = distance;
      //printf("%d(%d, %d) %f %f \n", bid, tidx, tidy, distance, distanceBackup);
      if(distance < distanceBackup)   //if new distance is lower than the old , reject.
      {
      	fitnessMatrix[tidx][tidy] = distance;
	//distanceBackup = distance;
      }
    }
    
     __syncthreads();

    Min = INT_MAX;
    //int currentMin = 0;
    
    if(threadIdx.x == 0 && threadIdx.y == 0) {
      for( i = 1 ; i < currentNumCities -1 ; i++ ) {
      	for ( j = i ; j < currentNumCities -1 ; j++) {
      	  improvement += fitnessMatrix[i][j];

      	  if (fitnessMatrix[i][j] < Min )
      	  {
      	    Min = fitnessMatrix[i][j];
      	    swapCities[0] = i;
      	    swapCities[1] = j;
      	  }
      	} 
      }
    

      temp = globalTourThreads[swapCities[0]];
      globalTourThreads[swapCities[0]] = globalTourThreads[swapCities[1]];
      globalTourThreads[swapCities[1]] = temp;
   
      if((swapCities[0] == 0 && swapCities[1] == 0) || counter ==  MAX_ITERATIONS - 1) {
	for(i = 0 ; i < currentNumCities; i++) {
	  completeTour[offset + i] = globalTourThreads[i];
	}
	//printf("In termination loop for block:%d, counter:%d\n", bid, counter);
	break;
      }

    }
    __syncthreads();

    distanceBackup = fitnessMatrix[swapCities[0]][swapCities[1]];
    fitnessMatrix[swapCities[0]][swapCities[1]] = INT_MAX;
  
    /*
     * find the max value from fitness and swap them
     * in the localtour and update localtour backup
     * for all threads.
     */
    counter++;
  }   
  /*
   * find the max value from fitness and swap only 
   * those elements in completeTour 
   */
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
    cudaMalloc(&coords_dMat,2 * n * sizeof(int));
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

void copyDMatToGPU(unsigned int n, int * Dest_gpu, const int* Src_cpu)
{

    cudaMemcpy(Dest_gpu , Src_cpu, 2 * n * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_GPU ("Copying coords to GPU");
}

void
copyKeysFromGPU (unsigned int n, int* Dest_cpu, int* Src_gpu)
{
  cudaMemcpy (Dest_cpu, Src_gpu, n * sizeof (int),cudaMemcpyDeviceToHost);  
  CHECK_GPU ("Copying keys from GPU");
}

extern "C"
void TSPSwapRun(int* tour,const int* coords)
{
  int n = NUM_CITIES;
  int* tour_gpu = createArrayOnGPU(n);
  
  int* coords_gpu = createDMatOnGPU(NUM_CITIES);
  int *newTour = (int *)malloc(sizeof(int) * n);

  int *coordsArray = (int *)malloc(2 * n * sizeof(int));
  for(int i = 0; i < n; i++)
  {
    coordsArray[2 * i] = coords[2 * i];
    coordsArray[(2 * i) + 1] = coords[(2 * i) + 1];
  }

  printf("CUDA PATH before KERNEL\n");
  for(int i = 0 ; i < NUM_CITIES; i++)
  {
     if(tour[i] == 0)
	printf("PANIC: zero city passed\n");
  }
 
  copyKeysToGPU (n,tour_gpu, tour);
  copyDMatToGPU(n, coords_gpu, coords);

  dim3 threadsPerBlock((NUM_CITIES/4) + 1, (NUM_CITIES/4) +1);

  TSPSwapKernel<<<NUM_BLOCKS_CUDA,threadsPerBlock>>> (n, tour_gpu, coords_gpu, localIter);
  cudaDeviceSynchronize();
  
  copyKeysFromGPU(n, tour, tour_gpu);

  printf("CUDA PATH after KERNEL\n");
  for(int i = 0 ; i < NUM_CITIES; i++)
	printf("%d-", tour[i]);
  printf("\n");
  if(tour_gpu)
    cudaFree(tour_gpu);
  
  if(coords_gpu)
    cudaFree(coords_gpu);
}

/*int main()
{
  int tour[NUM_CITIES] = {0, 2, 1 ,3, 4, 6,5, 7, 8, 10, 9, 11, 12, 14, 13, 15, 16};
  int* coords = (int*)malloc(2 * NUM_CITIES * sizeof(int *));
  
  for(int i = 0; i < NUM_CITIES; i++)
  {
    coords[(2 * i)] = i;
    coords[(2 * i) + 1] = i;
  }
  TSPSwapRun((int *)tour, (const int*)coords);
}*/
