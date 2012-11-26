#/bin/bash

rm -rf *.o
rm -rf TSP
#nvcc -c TSPCuda.cu -o TSPCuda.o -arch sm_21
#mpicc -g -o TSP driver.c readFromFile.c globalPopGen.c IndMPINode.c TSPCuda.o -I/usr/lib/openmpi/include/ -L/usr/lib/openmpi/include/ -L/opt/cuda-4.2/cuda/lib64 -fopenmp -lgomp -lmpi -lm -lcuda -lcudart
#mpicc -g -o TSP driver.c readFromFile.c globalPopGen.c  -I/usr/lib/openmpi/include/ -L/usr/lib/openmpi/include/ -L/opt/cuda-4.2/cuda/lib64 -lgomp -lmpi -lm -lcuda -lcudart
mpicc -g -o TSP driver.c readFromFile.c globalPopGen.c  -I/usr/lib/openmpi/include/ -L/usr/lib/openmpi/include/ -L/opt/cuda-4.2/cuda/lib64 -fopenmp -lgomp -lmpi -lm -lcuda -lcudart
#
