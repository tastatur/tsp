#include "globalData.h"
#include <stdint.h>
#include "readFromFile.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/** Generates number of tours each node should handle 
*   ith ranked node will handle tours from index tourCountOnNode[i-1]  to tourCountOnNode[i] - 1
**/

void findTourCountForNode(uint8_t *tourCountOnNode){
	uint8_t i , quot; 
	int tempCityCount = NUM_CITIES;
	
	quot = (uint8_t)ceil((float)tempCityCount/(numMPINodes - 1));
	tourCountOnNode[0] = quot;
	tempCityCount -= quot ;
	 
	for (i = 1 ; i < numMPINodes - 1 ; i++)
	{
		quot = (int)ceil((float)tempCityCount/(numMPINodes - i - 1));
		tourCountOnNode[i] = quot + tourCountOnNode[i-1];
		tempCityCount -= quot;
	}	
}
/**
* @Returns the fitness of the tour 	
**/
int computeFitness(int * tour){
	return 0;
}
int main(int argc , char **argv)
{
	uint8_t i, rowPerProc, startRow, gIter, lIter, j, tourCountOnNode[numMPINodes-1];
  	int **dMat;
  	int **TSPData_coordinates;
  	char *path = (char *)malloc(sizeof(char) * pathLen);

        /* Initialize the MPI communication world */
        int  rank, size;
        MPI_Status status;
	MPI_Request *requestD , *requestM, reqStatusD, reqStatusM;

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
		//MPI_Bcast(dMat , NUM_CITIES * NUM_CITIES , MPI_INT, 0,  MPI_COMM_WORLD);	
		
		/* Generate request handles */
		requestD = (MPI_Request *) malloc(sizeof(MPI_Request) * (size - 1));
		requestM = (MPI_Request *) malloc(sizeof(MPI_Request) * (size - 1));
		

		/******************************* This will be executed on MASTER for fixed number of global iterations ****************/
		for (gIter = 0 ; gIter < globalIter  ; gIter++){

			/* Distribute this global population across all MPI nodes */
			for (i = 0 ; i < size - 1 ; i++){
				rowPerProc = (i == 0) ? tourCountOnNode[i] : (tourCountOnNode[i] - tourCountOnNode[i-1]);
				startRow = (i == 0) ? 0 : tourCountOnNode[i - 1];

				for (j = 0 ; j < rowPerProc ; j++)
					MPI_Send(initialPopulation[startRow + j] ,  NUM_CITIES , MPI_INT, i + 1 , 1 , MPI_COMM_WORLD);
			}
			
			/* Wait to receive the best two tours from all worker MPI nodes */
			for (i = 0 ; i < size - 1 ; i++){
				startRow = (i == 0) ? 0 : tourCountOnNode[i - 1];
				MPI_Irecv(initialPopulation[startRow] , NUM_CITIES , MPI_INT , i+1 , i+1 , MPI_COMM_WORLD , &requestD[i]);
				MPI_Irecv(initialPopulation[startRow+1] , NUM_CITIES , MPI_INT , i+1, 100*(i+1)+1 , MPI_COMM_WORLD , &requestM[i]);
			}

			/* Wait till all nodes send their best solutions */
			MPI_Waitall(size - 1 , requestD, &status);
			MPI_Waitall(size - 1 , requestM, &status);
		
			/* Now improve these tours to generate a global population of improved tours */
		}
		/************************************************************************************************************/		

		/* At last we have the most optimized tour */
  		//resultVerification(TSPData_coordinates);
	}
	else{
		rowPerProc = ( rank == 1) ? tourCountOnNode[0] : (tourCountOnNode[rank - 1] - tourCountOnNode[rank - 2]);
		initialPopulation = (int **)malloc(sizeof(int *) * rowPerProc);

		for (i = 0 ; i < rowPerProc ; i++)
			initialPopulation[i] = (int *) malloc(sizeof(int) * NUM_CITIES);
	
		/******************************* This will be executed on WORKERS for fixed number of global iterations ****************/
		for (gIter = 0 ; gIter < globalIter  ; gIter++){
			/* Receive the distance matrix */
			//MPI_Bcast(dMat , NUM_CITIES * NUM_CITIES , MPI_INT, 0,  MPI_COMM_WORLD);	

			/* Receive the initial tours */
			for (i = 0 ; i < rowPerProc ; i++)
				MPI_Recv(initialPopulation[i] , NUM_CITIES, MPI_INT , 0 , 1 , MPI_COMM_WORLD , &status);

			/* Divide these tours among OpenMP threads */
			// ProcessRoute(initialPopulation,rowPerProc,TSPData_coordinates);

			/* Find two most optimal tour across all OpenMP threads on this MPI mode and send to MASTER */
			MPI_Isend(initialPopulation[0] , NUM_CITIES , MPI_INT , 0 , rank , MPI_COMM_WORLD , &reqStatusD);
			MPI_Isend(initialPopulation[1] , NUM_CITIES , MPI_INT , 0 , rank * 100 + 1 , MPI_COMM_WORLD , &reqStatusM);
						
			MPI_Waitall(1 , &reqStatusD , &status);
			MPI_Waitall(1 , &reqStatusM , &status);				
		}
	
		/************************************************************************************************************/		
	}
        MPI_Finalize();
	return 0;
}

