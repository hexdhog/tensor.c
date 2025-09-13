CC = gcc
CFLAGS = -Wall
SRCS = tensor.c

all: test

build:
	mkdir -p build

.PHONY: clean
clean:
	rm -rf build

.PHONY: test
test: build
	${CC} ${CFLAGS} ${SRCS} test.c -o build/test && ./build/test