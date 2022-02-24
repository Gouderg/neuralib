CC = g++
CFLAGS = -W -Wall -g
LDFLAGS = 

SRC = $(wildcard src/*.cpp)
OBJS = $(SRC:.cpp=.o)
HEADER = $(SRC:.cpp=.hpp)
 
all : neuralib clean

neuralib : $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o : %.c
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm src/*.o