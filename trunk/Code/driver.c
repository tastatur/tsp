#include "globalData.h"

#include <stdio.h>
#include <stdlib.h>

void main()
{
  int i,j;
  float **dMat;
  char *path = (char *)malloc(sizeof(char) * 100);
  dMat = (float **)malloc(sizeof(float *) * NUM_CITIES);
  for(i = 0 ; i < NUM_CITIES; i++)
    dMat[i] = (float *)malloc(sizeof(float) * NUM_CITIES);

  readDataFromFile(path, dMat);
  GenerateInitPopulation(dMat);
}


