C=mpicc
CFLAGS=-I/usr/lib/openmpi/include/ -c -Wall
LDFLAGS=-L/usr/lib/openmpi/include/
LIB=-lgomp -lmpi
SOURCES=driver.c readFromFile.c globalPopGen.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=TSP

all: $(SOURCES) $(EXECUTABLE) $(LIB)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@
