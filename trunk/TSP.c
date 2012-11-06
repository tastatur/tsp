/*
 * Author - Harshit Mehrotra
 * Project - Massively Parallel Genetic Travelling Salesman Algorithm
 * Course - CS6220
 */

# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <mpi.h>
# include <omp.h>

# define MAX_SUB_ITERATIONS 1000
# define MAX_GLOBAL_ITERATIONS 1000
# define GLOBAL_POP_SIZE 1008
# define NUM_THREADS 4

// Credits : Random tour generator using Fisher Yates shuffling algorithm is obtained from http://stackoverflow.com/questions/3343797/is-this-c-implementation-of-fisher-yates-shuffle-correct
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

void shuffle(int *array, int n) 
{
  int i, j, tmp;

  for (i = n - 1; i > 0; i--) 
  {
    j = rand_int(i + 1);
    tmp = array[j];
    array[j] = array[i];
    array[i] = tmp;
  }
}

// This function computes the fitness for given solution - has to be CUDA kernel
double computeFitness(int *subPop , int (*TSPData)[3] , int NUM_CITY)
{
	double distance = 0.0;
	int iLoop;

	for (iLoop = 1; iLoop < NUM_CITY ; iLoop++)
	{
		distance += sqrt( pow(TSPData[subPop[iLoop]][1] - TSPData[subPop[iLoop-1]][1] , 2) +  pow(TSPData[subPop[iLoop]][2] - TSPData[subPop[iLoop-1]][2] ,2)) ;
	}
	
	distance += sqrt( pow(TSPData[subPop[iLoop-1]][1] - TSPData[subPop[0]][1] , 2) +  pow(TSPData[subPop[iLoop-1]][2] - TSPData[subPop[0]][2] ,2)) ;
	return 1/distance; 
}

int main( int argc , char *argv[])
{
	int NUM_CITY = 0 , num , counter = 0 , size , iLoop, jLoop;
	char ch;
	
	// Initialize the MPI communicator
	int  rank, rc; 
	int temp , randA , randB , FLAG;
	int localGlobalIter = 0;
	double start , stop;
	MPI_Status status;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	start = MPI_Wtime();

	if (rank == 0)
	{
		// Read the city coordinates
		FILE *fin = fopen("./Project/TSP48.txt" , "r");
	
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
	      		
		fseek(fin , SEEK_SET , 0);
	
		int TSPData[NUM_CITY][3];
		
		while (fscanf(fin , "%d" , &num) != EOF)
		{
			TSPData[counter][0] = num;

			fscanf(fin , "%d" , &num);
			TSPData[counter][1] = num;

			fscanf(fin , "%d" , &num);
			TSPData[counter++][2] = num;		
		}
	
		fclose(fin);
	
		// Initialize the subpopulation
		int subPopulation[GLOBAL_POP_SIZE][NUM_CITY];
 
		for (iLoop = 0 ; iLoop < GLOBAL_POP_SIZE  ; iLoop++)
		{
			if (iLoop == 0)
			{	
				// Initialize the first set of population serialy
				for (jLoop = 0 ; jLoop < NUM_CITY ; jLoop++)
				{
					subPopulation[iLoop][jLoop] = jLoop + 1;
				}
				// Shuffle this population set
				shuffle(subPopulation[iLoop] , NUM_CITY) ;
			}
			else
			{
				//Initialize next set of population
				for (jLoop = 0 ; jLoop < NUM_CITY ; jLoop++)
				{
					subPopulation[iLoop][jLoop] = subPopulation[iLoop-1][jLoop];
				}
				// Shuffle this population set
				shuffle(subPopulation[iLoop] , NUM_CITY);
			}
		}
		
		int rowPerProc = GLOBAL_POP_SIZE / (size-1);
		int globalIter = 0;

		while (globalIter < MAX_GLOBAL_ITERATIONS)
		{
			for (iLoop = 1 ; iLoop < size ; iLoop++)
			{
				// Inform all processes about number of cities
				MPI_Send(&NUM_CITY , 1 , MPI_INT , iLoop , 0 , MPI_COMM_WORLD);
				// Now send iLoop th population set to process with rank iLoop+1
				MPI_Send(subPopulation[(iLoop - 1) * rowPerProc] , rowPerProc * NUM_CITY , MPI_INT, iLoop , 1 , MPI_COMM_WORLD);
				//Now send TSPData
				MPI_Send(TSPData, NUM_CITY * 3 , MPI_INT , iLoop , 2 , MPI_COMM_WORLD);
			}
			// All the nodes will now send their best solutions and rank 0 will breed these solutions to creater better solutions
			// Receive candidate solutions
			int recvdSubPop[size-1][NUM_CITY];
			int maxFit = 0;
			double maxFitness = 0 , oldFitnes = 0 , newFitnes = 0;

			for (iLoop = 1 ; iLoop < size ; iLoop++)
			{
				MPI_Recv(recvdSubPop[iLoop-1] , NUM_CITY , MPI_INT , iLoop , 4 , MPI_COMM_WORLD , &status);

				oldFitnes = computeFitness(recvdSubPop[iLoop-1] , TSPData , NUM_CITY);
				if (oldFitnes > maxFitness)
				{
					maxFitness = oldFitnes;
					maxFit = iLoop - 1;
				}
			}
			
			/*for (jLoop = 0 ; jLoop < NUM_CITY ; jLoop++)
			{
				printf(" %d" , recvdSubPop[maxFit][jLoop]);
			}
			
			printf(" Max Fitness : %0.15lf\n" , maxFitness);
			*/

			// Now swap best solution at maxFit to create even better solutions
			for (iLoop = 1 ; iLoop <= GLOBAL_POP_SIZE ; iLoop++)
			{
				do
				{
					FLAG = 1;
					randA = rand_int(NUM_CITY);
					randB = rand_int(NUM_CITY);
					
					if (randA < 1 || randA > NUM_CITY || randB < 1 || randB > NUM_CITY || randA == randB)
						FLAG = 0;	
					else
					{
						oldFitnes = computeFitness(recvdSubPop[maxFit] , TSPData , NUM_CITY);

						temp = recvdSubPop[maxFit][randA-1];
						recvdSubPop[maxFit][randA-1] = recvdSubPop[maxFit][randB-1];
						recvdSubPop[maxFit][randB-1] = temp;

						newFitnes = computeFitness(recvdSubPop[maxFit] , TSPData , NUM_CITY);
						if (newFitnes < oldFitnes)
						{
							FLAG = 0;
							// Swap back again as it is not an improvement
							temp = recvdSubPop[maxFit][randA-1];
							recvdSubPop[maxFit][randA-1] = recvdSubPop[maxFit][randB-1];
							recvdSubPop[maxFit][randB-1] = temp;
						}		
						else
						{
							FLAG = 0;
							// A good solution is created - copy it to subPopulation Buffers
							for (jLoop = 0 ; jLoop < NUM_CITY ; jLoop++)
								subPopulation[iLoop-1][jLoop] = recvdSubPop[maxFit][jLoop];
						}
						
					}	
				}while(FLAG == 0);
			}
			globalIter++;
			if (globalIter == MAX_GLOBAL_ITERATIONS)
			{
				for (jLoop = 0 ; jLoop < NUM_CITY ; jLoop++)
				{
					printf(" %d" , recvdSubPop[maxFit][jLoop]);
				}
			
				printf(" Max Fitness : %0.15lf\n" , maxFitness);
				stop = MPI_Wtime();
				printf("Execution Time : %0.15lf\n" , stop - start );
				printf("Iterations : %d" , MAX_GLOBAL_ITERATIONS * MAX_SUB_ITERATIONS );
				}
		}
	}
	else
	{
		
	while(localGlobalIter < MAX_GLOBAL_ITERATIONS)
	{
		// Receive the number of cities
		MPI_Recv(&NUM_CITY , 1, MPI_INT , 0 , 0 , MPI_COMM_WORLD , &status);
		
		//Initialize the local subpopulation buffer
		int rowPerProc = GLOBAL_POP_SIZE / (size-1);
		int subPop[rowPerProc][NUM_CITY];
		double FitnessScore[rowPerProc];		
		
		//Receive the subpopulation
		MPI_Recv(subPop , rowPerProc * NUM_CITY, MPI_INT , 0 , 1 , MPI_COMM_WORLD , &status);
		
		//Receive TSPData
		int TSPData[NUM_CITY][3];
		MPI_Recv(TSPData ,NUM_CITY * 3 , MPI_INT , 0 , 2 , MPI_COMM_WORLD , &status);
	
		int subIterCounter ;
		double oldFitness = 0 , newFitness = 0;
		
		omp_set_num_threads(NUM_THREADS);		

		// Split the subpopulation into four groups (threads) each thread having rowPerProc/4 number of chromosomes then find best 4 candidate solutions
		# pragma omp parallel for shared(rowPerProc , iLoop , subPop) private(subIterCounter, oldFitness , newFitness, randA, randB, temp, FLAG) schedule(static , 1) 
		for (iLoop = 0 ; iLoop < rowPerProc ; iLoop++)
		{	
			subIterCounter = 0;
			while (subIterCounter <= MAX_SUB_ITERATIONS)
			{
				// Perform a point-mutation by randomly selecting two points and swapping - accept the solution only if it increases the fitness
				oldFitness = computeFitness(subPop[iLoop] , TSPData , NUM_CITY);
				//printf("%d : %d" , iLoop , subIterCounter);

				do
				{
					FLAG = 1;
					randA = rand_int(NUM_CITY);
					randB = rand_int(NUM_CITY);
					
					if (randA < 1 || randA > NUM_CITY || randB < 1 || randB > NUM_CITY || randA == randB)
						FLAG = 0;	
				}while(FLAG == 0);
				
				temp = subPop[iLoop][randA-1];
				subPop[iLoop][randA-1] = subPop[iLoop][randB-1];
				subPop[iLoop][randB-1] = temp;

				newFitness = computeFitness(subPop[iLoop] , TSPData , NUM_CITY);
	
				if (newFitness < oldFitness)
				{
					//Swap back to original position
					temp = subPop[iLoop][randA-1];
					subPop[iLoop][randA-1] = subPop[iLoop][randB-1];
					subPop[iLoop][randB-1] = temp;
				}
				else
					subIterCounter++;
			}
		}
		// End of parallel section
		
		if (rank == 1)
		{
			/*for (iLoop = 0 ; iLoop  < rowPerProc ; iLoop++)
			{
				printf("\n");
				for(jLoop = 0 ; jLoop  < NUM_CITY ; jLoop++)
				{
					printf("%d " , subPop[iLoop][jLoop]);
				}
//				printf("AT Local : %lf" , computeFitness(subPop[iLoop] , TSPData , NUM_CITY) );
			}*/
		}
		
	
		// Compue the maximum fitness score
		double maxFitness = computeFitness(subPop[0] , TSPData , NUM_CITY);
		int maxFit = 0;
		
		for (iLoop = 1 ; iLoop < rowPerProc ; iLoop++)
		{
			oldFitness = computeFitness(subPop[iLoop] , TSPData , NUM_CITY);
			if (oldFitness > maxFitness)
			{		
				maxFitness = oldFitness;
				maxFit = iLoop;
			}
		}
	
		// Send the most fit chromosome to rank 0
		MPI_Send(subPop[maxFit], NUM_CITY , MPI_INT ,  0  , 4 , MPI_COMM_WORLD);
		localGlobalIter++;
		}
	}
//	stop = MPI_Wtime();
//	printf("Execution Time : %0.15lf" , stop - start );
	MPI_Finalize();
	return 0;
}

