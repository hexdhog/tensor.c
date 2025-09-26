#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <errno.h>
#include <limits.h>

typedef int32_t dim_t;
typedef int32_t dim_sz_t; // TODO: int32_t or int64_t?
typedef uint32_t stride_t;

// TODO: add view_t to tensor_t to allow multiple tensors to point to the same data (good idea?)
typedef struct {
    uint32_t refs; // number of tensors that point to this view (0 -> data should be freed)
    float *data;
} view_t;

// TODO: add strides
typedef struct {
    dim_t ndim;
    dim_sz_t *shape;
    stride_t *stride;
    uint32_t numel;
    float *data;
} tensor_t;

typedef enum {
    OP_ADD,
    OP_MUL
} tensor_op_t;

tensor_t *tensor_alloc(dim_t ndim, dim_sz_t *shape);
void tensor_free(tensor_t *t);
bool is_contiguous(tensor_t *t);
tensor_t *contiguous(tensor_t *t);
dim_t resolve_shape(uint32_t numel, dim_t ndim, dim_sz_t *shape);
dim_t resolve_dim(dim_t ndim, dim_t dim);
uint8_t broadcast(dim_t andim, dim_sz_t *asrc, dim_sz_t **adst, dim_t bndim, dim_sz_t *bsrc, dim_sz_t **bdst);
tensor_t *squeeze(tensor_t *t, dim_t dim);
tensor_t *unsqueeze(tensor_t *t, dim_t dim);
tensor_t *transpose(tensor_t *t, dim_t dim1, dim_t dim2);
tensor_t *reshape(tensor_t *t, dim_t ndim, dim_sz_t *shape);

tensor_t *min(tensor_t *t);
tensor_t *max(tensor_t *t);
tensor_t *sumall(tensor_t *t);
tensor_t *sum(tensor_t *t, int16_t dim, bool keepdim);
tensor_t *add(tensor_t *a, tensor_t *b);
tensor_t *mul(tensor_t *a, tensor_t *b);
void tfinfo(FILE *stream, tensor_t *t);
void tinfo(tensor_t *t);
void tfprint(FILE *stream, tensor_t *t);
void tprint(tensor_t *t);
void tprint_shape(uint32_t n, dim_sz_t *shape);
void tfprint_shape(FILE *stream, uint32_t n, dim_sz_t *shape);
void tprint_stride(uint32_t n, stride_t *stride);
void tfprint_stride(FILE *stream, uint32_t n, stride_t *stride);

#endif
