#include <omp.h>
//#ifndef globaldata
//#include "globalData.h"
//#endif

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
      int numLocalCities = 10;// BLOCKSIZE ;
      int numLocalCitiesFinal = 10;//NUM_CITIES - (3 * BLOCKSIZE);
  
//      TSPSwapRun(tour, coords);
    }
  }
}

