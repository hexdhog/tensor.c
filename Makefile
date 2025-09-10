CC = gcc
CFLAGS = -Wall
SRCS = tensor.c

all: test

build:
	mkdir -p build

.PHONY: test
test: build
	${CC} ${SRCS} test.c -o build/test && ./build/test