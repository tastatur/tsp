#ifndef _GLOBAL_H
#include "globalData.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/* Function declaration */
void printTour(int *t);
void CheckValidity(int *tour);
int GetNearestCity(int currCity, int** dMat, int* visited);
void GenerateTour(int initialCity, int* tourPointer, int** dMat);
int **GenerateInitPopulation(int **dMat);

/* Function definition */
int** GenerateInitPopulation(int** dMat)
{
  int i, city;

  int **initialPopulation = (int **)malloc(sizeof(int *) * NUM_CITIES);
  for(i = 0; i < NUM_CITIES; i++)
    initialPopulation[i] = (int *)malloc(sizeof(int) * NUM_CITIES);

  for(city = 0; city < NUM_CITIES; city++)
  {
    GenerateTour(city, initialPopulation[city], dMat);
    CheckValidity(initialPopulation[city]);
  }

  for(i = 0; i < NUM_CITIES; i++)
    printTour(initialPopulation[i]);

  return initialPopulation;
}


void printTour(int *t)
{
  int i;
  for(i = 0; i < NUM_CITIES; i++)
    printf("%d->", t[i]);
  printf("\n");
}

void GenerateTour(int initialCity, int* tourPointer, int** dMat)
{
  int i;
  int visited[NUM_CITIES];
  int currentCity;
  int nextCity;

  for(i = 0; i < NUM_CITIES; i++)
    visited[i] = 0;
  
  tourPointer[0] = initialCity;
  visited[initialCity] = 1;
  currentCity = initialCity;

  for(i = 1; i < NUM_CITIES; i++)
  {
    nextCity = GetNearestCity(currentCity, dMat, visited);
    tourPointer[i] = nextCity;
    currentCity = nextCity;
    visited[nextCity] = 1;
  }
}

int GetNearestCity(int currCity, int** dMat, int* visited)
{
  int i;
  int nextCity = INVALID;
  int distance = INT_MAX;
  for(i = 0 ; i < NUM_CITIES; i++)
  {
    if(dMat[currCity][i] < distance && visited[i] == 0)
    {
      distance = dMat[currCity][i];
      nextCity = i;
    }
  }
  
  if(nextCity == INVALID)
  {
    printf("ERROR:(GlobalPopGen1)Problem in tour generation\n");
    exit(0);    
  }
  return nextCity;
}

void CheckValidity(int *tour)
{
  int visited[NUM_CITIES];
  int i;
  
  for(i = 0; i < NUM_CITIES; i++)
    visited[i] = 0;

  for(i = 0; i < NUM_CITIES; i++)
  {
    if(visited[tour[i]] == 1)
    {
      printf("ERROR:Invalid path generated:1\n");
      exit(0);
    }
    visited[tour[i]] = 1;
  }

  for(i = 0; i < NUM_CITIES; i++)
  {
    if(visited[i] == 0)
    {
      printf("ERROR:Invalid path generated:2\n");
      exit(0);
    }
  } 
}