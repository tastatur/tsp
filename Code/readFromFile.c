#include "global.h"
#include "readFromFile.h"
#include<stdio.h>

int ** returnReadCitiesArray(char *path,int **TSPData)
{
	
	//int *TSPData_values;
	FILE *fin = fopen(path, "r");

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
	
	TSPData = (int *)malloc(sizeof(int)*NUM_CITY);

	int TSPData_values[NUM_CITY][3];
	//TSPData_values = (int *)malloc(sizeof(int)*NUM_CITY);

	for(int i=0;i<NUM_CITY;i++)
		TSPData[i] = (int *)malloc(sizeof(int)*NUM_CITY);


        while (fscanf(fin , "%d" , &num) != EOF)
                {
                        TSPData_values[counter][0] = num;

                        fscanf(fin , "%d" , &num);
                        TSPData_values[counter][1] = num;

                        fscanf(fin , "%d" , &num);
                        TSPData_values[counter++][2] = num;
                }

                fclose(fin);

}

int ** readActualPath(char *path)
{
        FILE *fin = fopen(path, "r");

        if (fin == NULL)
                {
                        printf("File not found");
                        getchar();
                        exit(1);
                }
	


}

int ** make2DArray(int **TSPData , int TSPData[][], int NUM_CITY)
{


}

/*open file in path - calculate length of file / number of cities - malloc size of 2D array - read file line by line - enter co-ordinates into array */


}
