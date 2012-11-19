//#ifndef _GLOBAL_H
#include "globalData.h"
//#endif
#include "readFromFile.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/** Generates number of tours each node should handle 
*   ith ranked node will handle tours from index tourCountOnNode[i-1]  to tourCountOnNode[i] - 1
**/

void findTourCountForNode(int *tourCountOnNode){
	int i , tempCityCount = NUM_CITIES;
	int quot;

	quot = (int)ceil((float)tempCityCount/(numMPINodes - 1));
	tourCountOnNode[0] = quot;
	tempCityCount -= quot ;
	 
	for (i = 1 ; i < numMPINodes - 1 ; i++)
	{
		quot = (int)ceil((float)tempCityCount/(numMPINodes - i - 1));
		tourCountOnNode[i] = quot + tourCountOnNode[i-1];
		tempCityCount -= quot;
	}	
}
int main(int argc , char **argv)
{
	int i, tourCountOnNode[numMPINodes-1], rowPerProc, startRow , gIter , lIter, j;
  	int **dMat;
  	int **TSPData_coordinates;
  	char *path = (char *)malloc(sizeof(char) * pathLen);

        /* Initialize the MPI communication world */
        int  rank, size;
        MPI_Status status;

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	/* Initialize the data matrix */
	int **initialPopulation;
	TSPData_coordinates = (int**)malloc(sizeof(int*) * NUM_CITIES);
	for(i=0 ; i< NUM_CITIES ; i++)
		TSPData_coordinates[i] = (int*)malloc(sizeof(int) * NUM_CITIES);

	dMat = (int **)malloc(sizeof(int *) * NUM_CITIES);
  	for(i = 0 ; i < NUM_CITIES; i++)
    		dMat[i] = (int *)malloc(sizeof(int) * NUM_CITIES);
	
	/* Find tour count handled by each mpi node */
	findTourCountForNode(tourCountOnNode);
			
  	if(rank == 0){
		/* Read the TSP City coordinates and populate the distance matrix */
  		TSPData_coordinates = readDataFromFile(path, dMat);
		
		/* Generate a global population using Nearest Neighbor algorithm */
  		initialPopulation = GenerateInitPopulation(dMat);
		
		/* Brodcast the distance matrix */
		MPI_Bcast(dMat , NUM_CITIES * NUM_CITIES , MPI_INT, 0,  MPI_COMM_WORLD);	

		/******************************* This will be executed on MASTER for fixed number of global iterations ****************/
		for (gIter = 0 ; gIter < globalIter ; gIter++){

			/* Distribute this global population across all MPI nodes */
			for (i = 0 ; i < size - 1 ; i++){
				rowPerProc = (i == 0) ? tourCountOnNode[i] : (tourCountOnNode[i] - tourCountOnNode[i-1]);
				startRow = (i == 0) ? 0 : tourCountOnNode[i - 1];

				for (j = 0 ; j < rowPerProc ; j++)
					MPI_Send(initialPopulation[startRow + j] ,  NUM_CITIES , MPI_INT, i + 1 , 1 , MPI_COMM_WORLD);
			}
			
			/* Wait to receive the best tour from all worker MPI nodes */
	
			/* Now improve these tours to generate a global population of improved tours */
		}
		/************************************************************************************************************/		

		/* At last we have the most optimized tour */
  		//resultVerification(TSPData_coordinates);
	}
	else{
		/******************************* This will be executed on WORKERS for fixed number of global iterations ****************/
		for (gIter = 0 ; gIter < globalIter ; gIter++){
			/* Receive the distance matrix */
			MPI_Bcast(dMat , NUM_CITIES * NUM_CITIES , MPI_INT, 0,  MPI_COMM_WORLD);	

			/* Receive the initial tours (Number of tours can be diferent for each MPI node if global population size is not multiple of number of worker nodes) */
			rowPerProc = ( rank == 1) ? tourCountOnNode[0] : (tourCountOnNode[rank - 1] - tourCountOnNode[rank - 2]);
			initialPopulation = (int **)malloc(sizeof(int *) * rowPerProc);

			for (i = 0 ; i < rowPerProc ; i++)
				initialPopulation[i] = (int *) malloc(sizeof(int) * NUM_CITIES);
			
			for (i = 0 ; i < rowPerProc ; i++)
				MPI_Recv(initialPopulation[i] , NUM_CITIES, MPI_INT , 0 , 1 , MPI_COMM_WORLD , &status);

			ProcessRoute(initialPopulation,rowPerProc,dMat);

			/* Divide these tours among OpenMP threads */

			/* Within each OpenMP thread divide each tour into 4 subparts and optimize each subpart on CUDA kernel */
	
			/* Combine optimized subparts from CUDA kernels into a complete tour */

			/* Find the most optimal tour across all OpenMP threads on this MPI mode and send this tour to MASTER */
		}
		/************************************************************************************************************/		
	}

        MPI_Finalize();
	return 0;
}

