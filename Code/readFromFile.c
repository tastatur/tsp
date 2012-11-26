#include "globalData.h"
#include "readFromFile.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int * readDataFromFile(char *path, unsigned int **TSPData)
{
	int city_index = 0, num , i;
	strcpy(path , pathString);

	FILE *fin = fopen(path, "r");
	int *TSPData_values;

        if (fin == NULL)
       	{
                 printf("File not found");
                 getchar();
                 exit(1);
        }

        /* Find the number of cities from data */
	TSPData_values = (int *)malloc(sizeof(int) * NUM_CITIES * 2);

        while (fscanf(fin , "%d" , &city_index) != EOF)
        {
              fscanf(fin , "%d" , &num);
              TSPData_values[2*(city_index-1)] = num;
		
              fscanf(fin , "%d" , &num);
              TSPData_values[2*city_index-1] = num;
        }

        fclose(fin);
	make2DArray(TSPData , TSPData_values );
	return TSPData_values;
}

void readActualPath(char *path, int* correctPath)
{
        FILE *fin = fopen(path, "r");
	int k = 0, num;

        if (fin == NULL)
        {
               printf("File not found");
               getchar();
               exit(1);
        }
	
        while (fscanf(fin , "%d" , &num) != EOF)
        {
               correctPath[k] = num;
	}
	fclose(fin);
}

void make2DArray(unsigned int **TSPData , int *TSPData_values)
{
	int k , i , CityIndex = 0 , eachCityIndex;
	
	for ( k = 0 ; k < NUM_CITIES ; k++) {
		eachCityIndex = 0;
		for ( i = 0 ; i<NUM_CITIES ; i++) {
			TSPData[k][i] = (((unsigned int)pow((TSPData_values[CityIndex+1] - TSPData_values[eachCityIndex+1]),2) + 
                 			  (unsigned int)pow((TSPData_values[CityIndex] - TSPData_values[eachCityIndex]),2)));

			eachCityIndex += 2;
		}
		CityIndex += 2;
	}
}

/*int main()
{
float **TSP;
int i,j;
char *path = (char*)malloc(sizeof(char)*100);
TSP = (float**)malloc(sizeof(float*)*50);
for ( j =0 ; j< 50 ; j++)
	TSP[j] = (float*)malloc(sizeof(float)*50);
readDataFromFile(path,TSP);
}*/

/*open file in path - calculate length of file / number of cities - malloc size of 2D array - read file line by line - enter co-ordinates into array */



