#include "globalData.h"
#include "readFromFile.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "timer.c"

/** Generates number of tours each node should handle 
*   ith ranked node will handle tours from index tourCountOnNode[i-1]  to tourCountOnNode[i] - 1
**/

int * GenerateInitPopulation(unsigned int ** dMat);

void findTourCountForNode(int *tourCountOnNode){
	int i , quot, tempCityCount = NUM_CITIES;
	
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

void printInitialPopulation(int * initialPopulation){
	int i , stop = NUM_CITIES * NUM_CITIES;
	for (i = 0 ; i < stop; i++){
		if (i % NUM_CITIES == 0 )
			printf("\n");
		printf(" %d" , initialPopulation[i]);
	}
}

/* Entry point of code */
int main(int argc , char **argv)
{
	int i, rowPerProc, startRow, gIter, lIter, j, tourCountOnNode[numMPINodes-1];
  	unsigned int **dMat;
  	int *TSPData_coordinates;
  	char *path = (char *)malloc(sizeof(char) * pathLen);
	struct stopwatch_t * timer = NULL;
	long double commTime;

        /* Initialize the MPI communication world */
        int  rank, size;
        MPI_Status status, * rStatus;
	MPI_Request *request , reqStatus;

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	/* Initialize the initial population  */
	int *initialPopulation;

	/* Find tour count handled by each mpi node */
	findTourCountForNode(tourCountOnNode);

	/* Timers */
	stopwatch_init();
	timer = stopwatch_create();
			
  	if(rank == 0){
		dMat = (unsigned int **)malloc(sizeof(unsigned int *) * NUM_CITIES);
	  	for(i = 0 ; i < NUM_CITIES; i++)
    			dMat[i] = (unsigned int *)malloc(sizeof(unsigned int) * NUM_CITIES);
	
		/* Read the TSP City coordinates and populate the distance matrix */
  		TSPData_coordinates = readDataFromFile(path, dMat);
		
		/* Generate a global population using Nearest Neighbor algorithm */
  		initialPopulation = GenerateInitPopulation(dMat);
		
		/* Brodcast the city coordinates matrix */
		MPI_Bcast(TSPData_coordinates , NUM_CITIES * 2 , MPI_INT, 0,  MPI_COMM_WORLD);	
		
		/* Generate request handles */
		request = (MPI_Request *) malloc(sizeof(MPI_Request) * (size - 1));
		rStatus = (MPI_Status *) malloc(sizeof(MPI_Status) * (size - 1));
	
		/******************************* This will be executed on MASTER for fixed number of global iterations ****************/
		for (gIter = 0 ; gIter < globalIter ; gIter++){
		
			/* Distribute this global population across all MPI nodes */
			stopwatch_start(timer);
			for (i = 0 ; i < size - 1 ; i++){
				rowPerProc = (i == 0) ? tourCountOnNode[i] : (tourCountOnNode[i] - tourCountOnNode[i-1]);
				startRow = (i == 0) ? 0 : tourCountOnNode[i - 1];

				MPI_Send(&initialPopulation[startRow*NUM_CITIES] ,  rowPerProc * NUM_CITIES , MPI_INT, i + 1 , 1 , MPI_COMM_WORLD);
			}
			commTime = stopwatch_stop(timer);
	
			printf("\nThroughput : %lf" , (double)(NUM_CITIES * NUM_CITIES * sizeof(int)) / ((double)1048576 * commTime));
			printf("\nTimer value is : %Lg" , commTime);

			/* Wait to receive the best two tours from all worker MPI nodes */
			for (i = 0 ; i < size - 1 ; i++){
				startRow = (i == 0) ? 0 : tourCountOnNode[i - 1];
				MPI_Irecv(&initialPopulation[startRow] , NUM_CITIES * 2 , MPI_INT , i+1 , i+1 , MPI_COMM_WORLD , &request[i]);
			}

			/* Wait till all nodes send their best solutions */
			MPI_Waitall(size - 1 , request, rStatus);
			
			/* Now improve these tours n parallel to generate a global population of improved tours */
			# pragma omp parallel default (shared) private(rowPerProc , startRow) shared(i, size, tourCountOnNode, initialPopulation) num_threads(size - 1)
			{ 
				for (i = 0 ; i < size - 1 ; i++)
				{	
					rowPerProc = (i == 0) ? tourCountOnNode[i] : (tourCountOnNode[i] - tourCountOnNode[i-1]);
					startRow = (i == 0) ? 0 : tourCountOnNode[i - 1];
					improveGlobalPopulation(initialPopulation , startRow , rowPerProc , dMat);
				}
			} /* End of parallel region */
	//		printInitialPopulation(initialPopulation);
	//		sleep(20);
		}
		/************************************************************************************************************/		

		/* At last we have the most optimized tour */
  		//resultVerification(TSPData_coordinates);
	}
	else{
		rowPerProc = ( rank == 1) ? tourCountOnNode[0] : (tourCountOnNode[rank - 1] - tourCountOnNode[rank - 2]);
		initialPopulation = (int *) malloc(sizeof(int ) * rowPerProc * NUM_CITIES); //single pointer changes
		TSPData_coordinates = (int*)malloc(sizeof(int) * NUM_CITIES * 2);

		/* Receive the coordinates */
		MPI_Bcast(TSPData_coordinates , NUM_CITIES * 2 , MPI_INT, 0,  MPI_COMM_WORLD);	
		
		/******************************* This will be executed on WORKERS for fixed number of global iterations ****************/
		for (gIter = 0 ; gIter < globalIter  ; gIter++){
			
			/* Receive the initial tours */
			MPI_Recv(initialPopulation , NUM_CITIES * rowPerProc , MPI_INT , 0 , 1 , MPI_COMM_WORLD , &status);
			
			/* Divide these tours among OpenMP threads */
			// ProcessRoute(initialPopulation,rowPerProc,TSPData_coordinates);

			/* Find two most optimal tour across all OpenMP threads on this MPI mode and send to MASTER */
			MPI_Isend(initialPopulation , NUM_CITIES * 2, MPI_INT , 0 , rank , MPI_COMM_WORLD , &reqStatus);
			MPI_Waitall(1 , &reqStatus , &status);
		}
	
		/************************************************************************************************************/		
	}
        MPI_Finalize();
	return 0;
}

