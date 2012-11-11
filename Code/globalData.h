#ifndef _GLOBAL_H
#define _GLOBAL_H


/* Global Variables */
#define NUM_CITIES 48                                 // Number of cities
#define pathLen 100					//length of path to read file from
typedef int tour[NUM_CITIES];                         // tour data structure
typedef float distanceMatrix[NUM_CITIES][NUM_CITIES]; // Matrix holding the distance

char *pathName;

#define pathString "TSPData.txt"
#define outPathComputed "TSPComputed.dat"
#define outPathActual "TSPActual.dat"
#define numOpenMPthreads_citypaths 10
#define numMPINodes 10
#define numOpenMPThreadsPerMPINode 10
#define numberOfDivisions  4
#define cudaThreadBlocks = 4
#define cudaThreadsPerBlock = 16

/******** ERROR CODES *********/

#define INVALID -1

#endif /* _GLOBAL_H */
