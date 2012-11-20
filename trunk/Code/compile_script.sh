nvcc -c TSPCuda.cu -o TSPCuda.o
mpicc -c -o TSP driver.c readFromFile.c globalPopGen.c IndMPINode.c TSPCuda.o -I/usr/lib/openmpi/include/ -L/usr/lib/openmpi/include/ -lgomp -lmpi -lm
