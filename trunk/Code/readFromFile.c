#include "globalData.h"
#include "readFromFile.h"
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>


int ** readDataFromFile(char *path,float **TSPData)
{
	
	//int *TSPData_values;
	int NUM_CITY = 0,counter = 0,num , i;
	char ch;
	strcpy(path , pathString);

	FILE *fin = fopen(path, "r");
	int **TSPData_values;

        if (fin == NULL)
        	{
                        printf("File not found");
                        getchar();
                        exit(1);
                }

                // Find the number of cities from data
        while((ch = getc(fin)) != EOF)
                {
                        if(ch == '\n')
                           NUM_CITY++;
                }
//        printf("Number of cities %d\n",NUM_CITY);

	fseek(fin , SEEK_SET , 0);	
	
	TSPData_values = (int **)malloc(sizeof(int*)*NUM_CITY);
	//TSPData_values = (int *)malloc(sizeof(int)*NUM_CITY);

	for(i=0;i<NUM_CITY;i++)
		TSPData_values[i] = (int *)malloc(sizeof(int)*3);


        while (fscanf(fin , "%d" , &num) != EOF)
                {
                        TSPData_values[counter][0] = num;

                        fscanf(fin , "%d" , &num);
                        TSPData_values[counter][1] = num;

                        fscanf(fin , "%d" , &num);
                        TSPData_values[counter][2] = num;
//		printf("TSP values %d %d %d \n" , TSPData_values[counter][0] , TSPData_values[counter][1] , TSPData_values[counter][2]);
		counter++;
                }

                fclose(fin);
	make2DArray(TSPData , TSPData_values , NUM_CITY);

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

int ** make2DArray(float **TSPData , int **TSPData_values, int NUM_CITY)
{
int k , i;
for ( k = 0 ; k < NUM_CITY ; k++) {

for ( i = 0 ; i<NUM_CITY ; i++) {
		TSPData[k][i] = (((float)pow((TSPData_values[k][2] - TSPData_values[i][2]),2) + (float)pow((TSPData_values[k][1] - TSPData_values[i][1]),2)));
//	printf("%f \t",TSPData[k][i]);
	}
//	printf("\n");
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



