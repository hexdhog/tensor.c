CC = gcc
CFLAGS = -Wall -fsanitize=address -g
SRCS = tensor.c

all: test

build:
	mkdir -p build

.PHONY: clean
clean:
	rm -rf build

.PHONY: test
test: build
	${CC} ${CFLAGS} ${SRCS} debug.c test.c -o build/test && ./build/test