#include <omp.h>
#include "globalData.h"
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>

void ProcessRoute(int*,int,const int*);

void ProcessRoute(int* localPopulation, int numberOfTours,const int* coords)
{
  int tour,distance = 0 , city , prevX = 0 , prevY = 0 , i = 0 , populationStartIndex;
  int firstMax = INT_MAX , secondMax = INT_MAX ;
  int numCities = NUM_CITIES;
  omp_set_num_threads(numberOfTours);
  int fitness[numberOfTours];
  
#pragma omp parallel default(shared) shared(localPopulation, numCities, numberOfTours , coords , fitness) private(tour,prevX , prevY , distance , i)
{
#pragma omp for
    for(tour = 0; tour < numberOfTours; tour++)
    {
    //  int numLocalCities = BLOCKSIZE ;
     // int numLocalCitiesFinal = NUM_CITIES - (3 * BLOCKSIZE);
  	printf("\nID IS %d" , tour);
    TSPSwapRun((int *)(localPopulation + numCities*tour) , (const int*)coords);

    prevX = (coords + (2 * (localPopulation + numCities*tour)[0]))[0]; 

    //coords[((localPopulation + numCities*tour)[0])][0]; //(coords + (2 * (localPopulation + numCities*tour)[0]))[0];

    prevY = (coords + (2 * (localPopulation + numCities*tour)[0]))[1];

    //coords[((localPopulation + numCities*tour)[0])][1]; //(coords + (2 * (localPopulation + numCities*tour)[0]))[1];

    for(i = 0; i < numCities; i++)
    {
      city = (localPopulation + numCities*tour)[i];
      printf ("City %d " , city);
      distance += (float)((coords + (city*2))[1] - prevY) * ((coords + (city*2))[1] - prevY)
                   + (float)((coords + (city*2))[0] - prevX) * ((coords + (city*2))[0] - prevX);
      prevX = (coords + (city*2))[0];
      prevY = (coords + (city*2))[1];
    }
	printf("\nfitness %d" , distance);
	fitness[tour] = distance;
    }
}

//	firstMax = 0 ;
//	secondMax = 0;

    for ( i = 0 ; i < numberOfTours ; i ++ )
	{
	printf("came in loop");
		if( fitness[i] < firstMax ) {
			secondMax = firstMax;
			firstMax = fitness[i];}

		else if (fitness[i] < secondMax)
			secondMax = fitness[i];
	}
	printf("firsMax %d " , firstMax);
	printf("secondMax %d " , secondMax);
	
    populationStartIndex = 0;
    for ( i = 0 ; i < numCities ; i++ )
	{
		(localPopulation + numCities*(populationStartIndex))[i] = (localPopulation + numCities*firstMax)[i];
		(localPopulation + numCities*(populationStartIndex + 1))[i] = (localPopulation + numCities*secondMax)[i];
	}

}


/*int main()
{
  int i;
  int tour[NUM_CITIES] = {0, 2, 1 ,3, 4, 6,5, 7, 8, 10, 9, 11, 12, 14, 13, 15 , 16};
  int** coords = (int**)malloc(NUM_CITIES * sizeof(int *));
  for(i = 0 ; i < NUM_CITIES; i++)
  {
    coords[i] = (int *)malloc(2 * sizeof(int));
  }

  for(i = 0; i < NUM_CITIES; i++)
  {
    coords[i][0] = i;
    coords[i][1] = i;
  }
  ProcessRoute((int*)tour ,1 ,  (const int**)coords);
  //TSPSwapRun((int *)tour, 5 , (const int**)coords);
}*/



//	ProcessRoute((int*)0,5,(int*)0);

