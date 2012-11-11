#include "globalData.h"

#include <stdio.h>

void ReadDataFromFile(distanceMatrix dMat);
void main()
{
  int i,j;
  float **dMat;
  char *path = (char *)malloc(sizeof(char) * 100);
  dMat = (float **)malloc(sizeof(float *) * NUM_CITIES);
  for(i = 0 ; i < NUM_CITIES; i++)
    dMat[i] = (float *)malloc(sizeof(float) * NUM_CITIES);

  readDataFromFile(path, dMat);
  /* for(i = 0; i < NUM_CITIES; i++) */
  /* { */
  /*   for(j =0 ; j < NUM_CITIES; j++) */
  /*   { */
  /*     if(dMat[i][j] == 0) */
  /* 	printf("Zero at %d %d\n",i,j); */
  /*   } */
  /* } */
  GenerateInitPopulation(dMat);
}


void ReadDataFromFile(distanceMatrix dMat)
{
  int i,j;

  for(i = 0; i < NUM_CITIES; i++){
    for(j = i; j < NUM_CITIES; j++)
    {
      dMat[i][j] = (float)((i * j) + 1);
      dMat[j][i] = dMat[i][j];
    }
  }


}
