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
//#define numOpenMPthreads_citypaths 10
#define numMPINodes 10
#define globalIter 1
#define localIter 1

#define NUM_BLOCKS = 4
#define BLOCKSIZE NUM_CITIES
//#define BLOCKSIZE(a) ((a) * (a))

/******** ERROR CODES *********/
#define INVALID -1
#define SUCCESS 1

#endif /* _GLOBAL_H */
