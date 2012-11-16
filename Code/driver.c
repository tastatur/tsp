#include "globalData.h"

#include <stdio.h>
#include <stdlib.h>
#ifndef MPI_1
#include<mpi.h>
#endif

void main()
{
  int i,j;
  float **dMat;
  int **TSPData_coordinates;
  char *path = (char *)malloc(sizeof(char) * 100);

  	int NUM_CITY = 0 , num , counter = 0 , size , iLoop, jLoop;
        char ch;

        // Initialize the MPI communicator
        int  rank, rc;
        int temp , randA , randB , FLAG;
        int localGlobalIter = 0;
        double start , stop;
        MPI_Status status;

        MPI_Init(&argc,&argv);
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        start = MPI_Wtime();

  dMat = (float **)malloc(sizeof(float *) * NUM_CITIES);
  for(i = 0 ; i < NUM_CITIES; i++)
    dMat[i] = (float *)malloc(sizeof(float) * NUM_CITIES);


  if(rank == 0){
  TSPData_coordinates = readDataFromFile(path, dMat);
  GenerateInitPopulation(dMat);
  resultVerification(TSPData_coordinates);
}
          MPI_Finalize();
return;
}

