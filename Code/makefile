C=gcc
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=driver.c readFromFile.c globalPopGen.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=TSP

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@
