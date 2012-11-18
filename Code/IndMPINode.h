#include "globalData.h"

void ProcessRoute(int** localPopulation, int numberOfTours);

const unsigned int BLOCKSIZE = NUM_CITIES * NUM_CITIES ;  /* threads per block */
const unsigned int NUM_BLOCKS = 4;      

