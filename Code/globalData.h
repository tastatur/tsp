
/* Global Variables */
#define NUM_CITIES 48                                 // Number of cities
typedef int tour[NUM_CITIES];                         // tour data structure
typedef float distanceMatrix[NUM_CITIES][NUM_CITIES]; // Matrix holding the distance

char *pathName;

const int numOpenMPthreads_citypaths = 10;
const int numMPINodes = 10;
const int numOpenMPThreadsPerMPINode = 10;
const int numberOfDivisions = 4;
const int cudaThreadBlocks = 4;
const int cudaThreadsPerBlock = 16;




