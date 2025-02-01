CC = g++
CFLAGS = -std=c++11 -pthread -fopenmp
TARGET = main
SRC = main.cpp

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC) $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(TARGET)