#include <omp.h>
#include "globalData.h"

/* Function prototype */
void ProcessRoute(int** localPopulation , int numberOfTours, int **coords);


/* Function Definition */
void ProcessRoute(int** localPopulation, int numberOfTours, int** coords)
{
  int tour;
  int blockSize;
  omp_set_num_threads(numberOfTours);
  

  #pragma omp parallel
  {
    #pragma omp default(none) shared(localPopulation, NUM_CITIES, numberOfTours) private(tour)
    for(tour = 0; tour < numberOfTours; tour++)
    {
      int numLocalCities =  BLOCKSIZE ;
      int numLocalCitiesFinal = NUM_CITIES - (3 * BLOCKSIZE);
  
//      TSPSwapRun(tour, coords);
    }
  }
}

