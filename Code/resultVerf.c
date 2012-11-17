#include<stdio.h>
#include "global.h"
#include "readFromFile.h"

void resultVerification(int *TSPDataComputed, int **TSPData_values)
{

int *correctPath ;
int x_cor , y_cor ;
const char *outfile;
const char *outfile_2;
returnActualPath(path,&correctPath);

//for( int i=0;i<NUM_CITY;i++)
//{
	//calculate fitness and find difference .
	
	strcpy(outfile,outPathActual);
	strcpy(outfile_2,outPathComputed);

	FILE *fp = fopen(outfile,"w");
	FILE *fp_2 = fopen(outfile_2,"w");
	for (i=0; i < NUM_CITY ;i++)
	{
	x_cor = TSPData_values[correctPath[i]][1];
	y_cor = TSPData_values[correctpath[i]][2];
	fprintf (fp, "%d\t%d\n",x_cor , y_cor);
	x_cor = TSPData_values[TSPDataComputed[i]][1];
	y_cor = TSPData_values[TSPDataComputed[i]][2];
	fprintf (fp_2 , "%d\t%d",x_cor,y_cor);
	}

	fflush (fp);
	fflush (fp_2);


	//write into finalOutput.dat the data values.

	
	//plot the 2 different data sets.
}
