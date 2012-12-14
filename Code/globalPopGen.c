#include "globalData.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <x86intrin.h>
/* Function declaration */
/* void printTour(int *t);
void CheckValidity(int *tour, char* text);
int GetNearestCity(int currCity, unsigned int** dMat, int* visited);
void GenerateTour(int initialCity, int* tourPointer, unsigned int** dMat);
int * GenerateInitPopulation(unsigned int **dMat);
void improveGlobalPopulation(int *, int, int, unsigned int**);
double computeFitness(int * , unsigned int **);
*/

double computeFitness(int *tour , unsigned int ** dMat){
	int i,j;
	double distance = 0.0;
	
	for (i = 0 ; i < NUM_CITIES - 1 ; i++){
		distance += dMat[tour[i] - 1][tour[i+1] - 1];
		}
	distance += dMat[tour[0] - 1][tour[NUM_CITIES - 1] - 1];

//	printf("\nDistance sequential is : %lf" , distance);

	/* Below code uses SIMD instructions to compute tour fitness which we use for experiments. However due to random acess to dMat matrix the performance deteriorates and we do not use it in system evaluation */
	/* int a[4] , b[4];
	__m128i sumOne;
	__m128i sumTwo;
	__m128i result;

	sumOne = _mm_setzero_si128();
	sumTwo = _mm_setzero_si128();
	result = _mm_setzero_si128();

	distance = 0.0;

	for (i = NUM_CITIES - 1; i >= 8 ; i=i-8)
	{
		a[0] = dMat[tour[i] - 1][tour[i-1] - 1];	
		a[1] = dMat[tour[i-1] - 1][tour[i-2] - 1];
		a[2] = dMat[tour[i-2] - 1][tour[i-3] - 1];
		a[3] = dMat[tour[i-3] - 1][tour[i-4] - 1];
	
		b[0] = dMat[tour[i-4] - 1][tour[i-5] - 1];	
		b[1] = dMat[tour[i-5] - 1][tour[i-6] - 1];
		b[2] = dMat[tour[i-6] - 1][tour[i-7] - 1];
		b[3] = dMat[tour[i-7 ] - 1][tour[i-8] - 1];

		sumOne = _mm_loadu_si128(a);
		sumTwo = _mm_loadu_si128(b);

		result = _mm_hadd_epi32(sumOne, sumTwo);
		distance += _mm_extract_epi32(result,0);
		distance += _mm_extract_epi32(result,1); 
		distance += _mm_extract_epi32(result,2);
		distance += _mm_extract_epi32(result,3);

		printf("result[0] = %lf" , _mm_extract_epi32(result,0));
		printf("distance %lf" , distance);
	} 

	for ( j = 0 ; j < i ; j++ ) {
		distance += dMat[tour[j] - 1][tour[j+1] - 1]; }
	
	distance += dMat[tour[0] - 1][tour[NUM_CITIES - 1] - 1];
	printf("\nDistance SIMD is : %lf" , distance); */
	/* ************************************* SIMD Intrinsics End ****************************************/
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

  for(city ; city < NUM_CITIES; city++)
  {
     GenerateTour(city + 1, &initialPopulation[city  * NUM_CITIES], dMat);
     CheckValidity(&initialPopulation[city  * NUM_CITIES] , "init");
  }

  return initialPopulation;
}

void printTour(int *t)
{
  int i;
  printf("\n");

  for(i = 0; i < NUM_CITIES; i++)
  	printf("%d-", t[i]);
}

/* void showBackTrace() 
{ 
  const int maxbtsize = 50; 
  int btsize;
  void* bt[maxbtsize];
  char** strs = 0; 
  int i = 0; 

  btsize = backtrace(bt, maxbtsize); 
  strs = backtrace_symbols(bt, btsize);

  for (i = 0; i < btsize; i += 1)
  {
  	printf("%d.) %s\n", i, strs[i]); 
  }

  free(strs);  
}*/

void GenerateTour(int initialCity, int* tourPointer, unsigned int** dMat)
{
  int i;
  int visited[NUM_CITIES + 1];
  int currentCity , nextCity;

  for(i = 0; i <= NUM_CITIES; i++)
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
    if(dMat[currCity - 1][i] < distance && visited[i + 1] == 0)
    {
      distance = dMat[currCity - 1][i];
      nextCity = i + 1;
    }
  }
  
  if(nextCity == INVALID)
  {
    printf("ERROR:(GlobalPopGen1)Problem in tour generation\n");
    exit(0);    
  }
  return nextCity;
}

void CheckValidity(int *tour, char *text)
{
  int visited[NUM_CITIES + 1];
  int i;
  
  for(i = 0; i <= NUM_CITIES; i++)
    visited[i] = 0;

  for(i = 0; i < NUM_CITIES; i++)
  {
    if(visited[tour[i]] == 1)
    {
      printf("ERROR:Invalid path generated:<%s>,city %d repeated\n" ,text, tour[i] );
      printTour(tour);
      exit(0);
    }
    visited[tour[i]] = 1;
  }

  for(i = 1; i <= NUM_CITIES; i++)
  {
    if(visited[i] == 0)
    {
      printf("ERROR:Invalid path generated:<%s>, city %d not present\n", text, i);
      printTour(tour);
      exit(0);
    }
  } 
}

void improveGlobalPopulation(int * initialPopulation , int startRow , int offSpringCount , unsigned int **dMat){
	int i , k;
	double dadFitness, tourFitness;

	struct timeval globalTime;
	/* Use two most fittest solution to generate children */
	int * dad = (int *)malloc(sizeof(int) * NUM_CITIES);
	int * mom = (int *)malloc(sizeof(int) * NUM_CITIES);

	/* Temporary Structures */
	int ** firstPath, ** secondPath;

	char ch = 'u';

	firstPath = (int **)malloc(sizeof(int *) * (NUM_CITIES + 1));
	secondPath = (int **)malloc(sizeof(int *) * (NUM_CITIES + 1));

	int * globalPopPool = (int *)malloc(sizeof(int) * NUM_CITIES);
	char * status = (char * )malloc(sizeof(char) * (NUM_CITIES + 1));

	int curLoc , flag, offset , pos , temp , buf;
	unsigned int distanceMin ;

	for (i = 0 ; i <= NUM_CITIES ; i++){
		firstPath[i] = (int *)malloc(sizeof(int) * 2);
		firstPath[i][0] = firstPath[0][1] = -1;

		secondPath[i] = (int *)malloc(sizeof(int) * 2);
		secondPath[i][0] = secondPath[i][1] = -1;
	}	
	/* Make temporary copy of most fit solutions */
	memcpy(dad , &initialPopulation[startRow * NUM_CITIES] , NUM_CITIES * sizeof(int)) ;
	memcpy(mom , &initialPopulation[(startRow  + 1 ) * NUM_CITIES] , NUM_CITIES * sizeof(int)) ;	

	CheckValidity(dad , "Dad");
	CheckValidity(mom , "Mom");

	dadFitness = computeFitness(dad , dMat);
//	printf("Dad");		printTour(dad); 	printf("\t");		
	gettimeofday(&globalTime, 0);	printf("Dad Fitness : %0.2lf , Global Time %ld\n  " , dadFitness, globalTime.tv_usec);
//	printf("Mom");		printTour(mom);		printf("\t");		
	gettimeofday(&globalTime, 0);	printf("Mom Fitness : %0.2lf , Global Time %ld\n  " , computeFitness(mom , dMat), globalTime.tv_usec);

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

	/* 	
	for (i = 1 ; i <= NUM_CITIES ; i++){
		printf("\n\t%d)\t%d %d \t\t%d %d", i , firstPath[i][0] , firstPath[i][1]  ,secondPath[i][0] , secondPath[i][1]);
	} */
	
	curLoc = 0;
	flag = 0;

	for (i = 0 ; i < NUM_CITIES ; i++)
	{
		if ( ( firstPath[dad[i]][0] != -1) && 
		     ((firstPath[dad[i]][0] == secondPath[dad[i]][0]) || (firstPath[dad[i]][0] == secondPath[dad[i]][1])) )	
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
		for (k = 0 ; k <= NUM_CITIES ; k++)
			status[k] = 'u';

		initialPopulation[(startRow + i) * NUM_CITIES] = globalPopPool[rand_int(curLoc)];
		offset = 1;
		temp = initialPopulation[(startRow + i) * NUM_CITIES] ;
		status[temp] = 'v';
		
		while(offset < NUM_CITIES) {	
		        if ( (firstPath[temp][0] != -1 ) && 
			     (status[firstPath[temp][0]] == 'u') && 
			     (( firstPath[temp][0] == secondPath[temp][0]) || (firstPath[temp][0] == secondPath[temp][1]) ))
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
					buf = dMat[temp-1][globalPopPool[k]-1];
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
		
		/******************************************** Tour Statistics *******************************/
		CheckValidity(&initialPopulation[(startRow + i) * NUM_CITIES] , "New population Generation");
		
		tourFitness = computeFitness(&initialPopulation[(startRow + i) * NUM_CITIES] , dMat);
		if (tourFitness < dadFitness){
			memcpy(&initialPopulation[(startRow + i) * NUM_CITIES] , dad , NUM_CITIES * sizeof(int));
			//printf("\nTour %d : " , i);		
			//tourFitness = computeFitness(&initialPopulation[(startRow + i) * NUM_CITIES] , dMat);
//			printTour(&initialPopulation[(startRow + i) * NUM_CITIES]);
			//gettimeofday( &globalTime, 0 );	
			//printf("\n\tFitness : %0.2lf , Global Time Improved Fitness %ld  " , tourFitness, globalTime.tv_usec);
		}
		//else /* Reject this new tour because it is less fitter than the parent */
		//	i--;
		
		/********************************************************************************************/
	}
	  	
	free(firstPath);
	free(secondPath);	
	free(globalPopPool);
	free(status);	
	free(dad);
	free(mom);
}

/*int main(int argc , char ** argv){
	unsigned int ** dMat	;
	int * initialPopulation;
	int i;
	struct stopwatch_t * timer = NULL;
	double runTime = 0.0;
	stopwatch_init();
	timer = stopwatch_create();	

	dMat = (unsigned int **)malloc(sizeof(unsigned int *) * NUM_CITIES);
        for(i = 0 ; i < NUM_CITIES; i++)
                dMat[i] = (unsigned int *)malloc(sizeof(unsigned int) * NUM_CITIES);

	char * path = (char * )malloc(sizeof(char) * 100);
	readDataFromFile(path , dMat);

	initialPopulation = GenerateInitPopulation(dMat);
	stopwatch_start(timer);

	for(i=0;i<100;i++)
	computeFitness(initialPopulation , dMat);

	runTime = stopwatch_stop(timer);

	printf("running time is : %lf " , runTime);

//	 improveGlobalPopulation(initialPopulation , 0 , 6 , dMat);	
	for ( i = 0 ; i < NUM_CITIES*6 ; i++ ) {
		if(i%NUM_CITIES == 0)
			printf("\n");
		printf("%d " , initialPopulation[i]);
	}
}*/
