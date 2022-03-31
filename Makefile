all: main.exe 

CC=g++ -std=c++17
CFLAGS= -O3 -march=native -fopenmp

%.o: src/%.cpp inc/%.hpp
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

main.exe: src/main.cpp ANN.o MNIST.o stopwatch.o
	$(CC) $(CFLAGS) $< -o $@ ANN.o MNIST.o stopwatch.o

# test: main.exe
# 	valgrind ./$< verify

test: main.exe
	./$< verify >>log
mnist: main.exe
	./$< mnist 

clean: 
	rm *o *exe log