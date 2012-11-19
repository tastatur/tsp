mpicc -o TSP driver.c readFromFile.c globalPopGen.c IndMPINode.c -I/usr/lib/openmpi/include/ -L/usr/lib/openmpi/include/ -lgomp -lmpi -lm
