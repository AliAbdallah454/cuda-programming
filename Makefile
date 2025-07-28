GCC = gcc
CFLAGS = -Wall -g

SRC = main.c utils.c
TARGET = myprogram

main: main.c
	$(GCC) main.c -o main

clean:
	rm -f main