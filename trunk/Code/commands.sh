qsub -I -q class -l nodes=1:gpu -d $(pwd)
nvcc -g -G TSPCuda.cu -o TSP -arch sm_21
cuda-gdb TSP
b //breakpointnumber
thread x,y (switch to thread id x,y for debugging)
mpirun --hostfile $PBS_NODEFILE -np 9 ./TSP
