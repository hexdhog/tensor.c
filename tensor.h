#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

typedef int32_t dim_t;
typedef uint32_t shape_t;
typedef uint32_t stride_t;

// TODO: add strides
typedef struct {
    dim_t ndim;
    shape_t *shape;
    stride_t *stride;
    uint32_t nelem;
    float *data;
} tensor_t;

typedef enum {
    OP_ADD,
    OP_MUL
} tensor_op_t;

tensor_t *tensor_alloc(dim_t ndim, shape_t *shape);
void tensor_free(tensor_t *t);
uint8_t broadcast(tensor_t *a, shape_t **ashape, tensor_t *b, shape_t **bshape);
tensor_t *squeeze(tensor_t *t, dim_t dim);
tensor_t *unsqueeze(tensor_t *t, dim_t dim);
tensor_t *transpose(tensor_t *t, dim_t dim1, dim_t dim2);
tensor_t *reshape(tensor_t *t, dim_t ndim, shape_t *shape);
tensor_t *min(tensor_t *t);
tensor_t *max(tensor_t *t);
tensor_t *sumall(tensor_t *t);
tensor_t *sum(tensor_t *t, int16_t dim, bool keepdim);
tensor_t *add(tensor_t *a, tensor_t *b);
tensor_t *mul(tensor_t *a, tensor_t *b);
void fprint(FILE *stream, tensor_t *t);
void print(tensor_t *t);
void print_shape(uint8_t ndim, uint16_t *shape);
void fprint_shape(FILE *stream, uint8_t ndim, uint16_t *shape);

#endif
