#include "globalData.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include<string.h>

/* Function declaration */
void printTour(int *t);
void CheckValidity(int *tour);
int GetNearestCity(int currCity, unsigned int** dMat, int* visited);
void GenerateTour(int initialCity, int* tourPointer, unsigned int** dMat);
int * GenerateInitPopulation(unsigned int **dMat);
void improveGlobalPopulation(int *, int, int, unsigned int**);

double computeFitness(int *tour , unsigned int ** dMat){
	int i;
	double distance = 0.0;
	
	for (i = 0 ; i < NUM_CITIES - 1 ; i++)
		distance += dMat[tour[i]][tour[i+1]];
	distance += dMat[tour[0]][tour[NUM_CITIES - 1]];
	printf("%lf" , distance);
	return distance;
}

/* Random number generator */
static int rand_int(int n) 

{
  int limit = RAND_MAX - RAND_MAX % n;
  int rnd;

  do 
  {
    rnd = rand();
  }while (rnd >= limit);

  return rnd % n;
}

/* Function definition */
int * GenerateInitPopulation(unsigned int** dMat)
{
  int i, city;

  int *initialPopulation = (int *)malloc(sizeof(int) * NUM_CITIES * NUM_CITIES) ;

  for(city = 0; city < NUM_CITIES; city++)
  {
     GenerateTour(city, &initialPopulation[city * NUM_CITIES], dMat);
     CheckValidity(&initialPopulation[city * NUM_CITIES]);
  }

  for(i = 0; i < 2 /* NUM_CITIES */; i++)
    printTour(&initialPopulation[i * NUM_CITIES]);

  return initialPopulation;
}

void printTour(int *t)
{
  int i;
  for(i = 0; i < NUM_CITIES; i++)
    printf("%d-", t[i]);
  printf("\n");
}

void GenerateTour(int initialCity, int* tourPointer, unsigned int** dMat)
{
  int i;
  int visited[NUM_CITIES];
  int currentCity , nextCity;

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

int GetNearestCity(int currCity, unsigned int** dMat, int* visited)
{
  int i;
  int nextCity = INVALID;
  unsigned int distance = INT_MAX;
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

void improveGlobalPopulation(int * initialPopulation , int startRow , int offSpringCount , unsigned int **dMat){
	int i , k;

	/* Use two most fittest solution to generate children */
	int * dad = (int *)malloc(sizeof(int) * NUM_CITIES);
	int * mom = (int *)malloc(sizeof(int) * NUM_CITIES);

	/* Temporary Structures */
	int ** firstPath, ** secondPath;

	char ch = 'u';

	firstPath = (int **)malloc(sizeof(int *) * NUM_CITIES);
	secondPath = (int **)malloc(sizeof(int *) * NUM_CITIES);

	int * globalPopPool = (int *)malloc(sizeof(int) * NUM_CITIES);
	char * status = (char * )malloc(sizeof(char) * NUM_CITIES + 1);

	int curLoc , flag, offset , pos , temp , buf;
	unsigned int distanceMin ;

	for (i = 0 ; i < NUM_CITIES ; i++){
		firstPath[i] = (int *)malloc(sizeof(int) * 2);
		secondPath[i] = (int *)malloc(sizeof(int) * 2);
	}
	
	/* Make temporary copy of most fit solutions */
	memcpy(dad , &initialPopulation[startRow * NUM_CITIES] , NUM_CITIES * sizeof(int)) ;
	memcpy(mom , &initialPopulation[(startRow  + 1 ) * NUM_CITIES] , NUM_CITIES * sizeof(int)) ;

	/* Special cases */	
	firstPath[dad[0]][1] = -1;
	secondPath[mom[0]][1] = -1;
	firstPath[dad[0]][0] = dad[1];
	secondPath[mom[0]][0] = mom[1];

	firstPath[dad[NUM_CITIES - 1]][0] = -1;
	secondPath[mom[NUM_CITIES - 1]][0] = -1;
	firstPath[dad[NUM_CITIES - 1]][1] = dad[NUM_CITIES - 2];
	secondPath[mom[NUM_CITIES - 1]][1] = mom[NUM_CITIES - 2];

	for (i = 1 ; i < NUM_CITIES - 1 ; i++){
		firstPath[dad[i]][0] = dad[i+1];
		secondPath[mom[i]][0] = mom[i+1];
		firstPath[dad[i]][1] = dad[i-1];
		secondPath[mom[i]][1] = mom[i-1];
	}
	
	/* for (i = 0 ; i < NUM_CITIES ; i++){
		printf("\n\t%d)\t%d %d \t\t%d %d", i , firstPath[i][0] , firstPath[i][1]  ,secondPath[i][0] , secondPath[i][1]);
	}*/
	
	curLoc = 0;
	flag = 0;

	for (i = 0 ; i < NUM_CITIES ; i++)
	{
		if ( (firstPath[dad[i]][0] == secondPath[dad[i]][0]) || (firstPath[dad[i]][0] == secondPath[dad[i]][1]) )	
		{
			if (!flag){
				globalPopPool[curLoc++] = dad[i];
				flag = 1;
			}
		}
		else{
			globalPopPool[curLoc++] = dad[i];
			flag = 0;
		}
	}

	for (i = 0 ; i < offSpringCount ; i++) {
		/* To optimize */
		for (k = 0 ; k < NUM_CITIES ; k++)
			status[k] = 'u';

		initialPopulation[(startRow + i) * NUM_CITIES] = globalPopPool[rand_int(curLoc)];
		offset = 1;
		temp = initialPopulation[(startRow + i) * NUM_CITIES];
		status[temp] = 'v';
		
		while(offset < NUM_CITIES) {	
			if ( (status[firstPath[temp][0]] == 'u') && ((firstPath[temp][0] == secondPath[temp][0]) || (firstPath[temp][0] == secondPath[temp][1]) ))
                	{
                                initialPopulation[(startRow + i) * NUM_CITIES + offset] = firstPath[temp][0]; 
				temp = firstPath[temp][0];
				status[temp] = 'v';
                	}
			else
			{
				/* find nearest element from current city */
				distanceMin = INT_MAX;
				pos = 0;
				for ( k = 0 ; k < curLoc ; k++ ) {
					buf = dMat[temp][globalPopPool[k]];
					if(status[globalPopPool[k]] == 'u' && buf < distanceMin) {
						distanceMin = buf;
						pos = k;
					}
				}
				
				initialPopulation[(startRow + i) * NUM_CITIES + offset] = globalPopPool[pos];
				temp = globalPopPool[pos];
				status[temp] = 'v';
			}	
			offset += 1;
		}
	}
	
	printTour(dad); printf("  ");computeFitness(dad , dMat);
	printf("\n");
	printTour(mom);  printf("  ") ; computeFitness(mom , dMat);

	for (i = 0 ; i < offSpringCount ; i++)
	{
		printTour(&initialPopulation[(startRow + i) * NUM_CITIES]); printf("   "); computeFitness(&initialPopulation[(startRow + i) * NUM_CITIES] , dMat);
		printf("\n");
	}
	free(firstPath);
	free(secondPath);	
	
	free(dad);
	free(mom);
}

/*int main(int argc , char ** argv){
	unsigned int ** dMat	;
	int * initialPopulation;
	int i;
	dMat = (unsigned int **)malloc(sizeof(unsigned int *) * NUM_CITIES);
        for(i = 0 ; i < NUM_CITIES; i++)
                dMat[i] = (unsigned int *)malloc(sizeof(unsigned int) * NUM_CITIES);

	char * path = (char * )malloc(sizeof(char) * 100);
	readDataFromFile(path , dMat);

	initialPopulation = GenerateInitPopulation(dMat);

	improveGlobalPopulation(initialPopulation , 0 , 6 , dMat);	
	for ( i = 0 ; i < NUM_CITIES*6 ; i++ ) {
		if(i%NUM_CITIES == 0)
			printf("\n");
		printf("%d " , initialPopulation[i]);
		}
		
}*/
