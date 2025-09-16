#include "tensor.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define RESOLVE_DIM(dim, ndim) ((dim_t)((dim) >= 0 ? (dim) : (dim) + (ndim)))
#define SWAP(a, b) do { typeof(a) tmp = (a); (a) = (b); (b) = tmp; } while (0)

/**
 * Creates a tensor and allocates the required memory for it
 * 
 * @param ndim number of dimensions for the tensor (number of elements in the shape argument)
 * @param shape shape of the tensor
 * @return pointer to the created tensor
 */
tensor_t *tensor_alloc(dim_t ndim, dim_sz_t *shape) {
    assert(ndim > 0); // TODO: should tensors be allowed to have 0 dimensions (for single values)?
    assert(shape != NULL);

    // shapes can contain negative numbers (for API pruposes; e.g. reshape(t, 3, -1, 2))
    // because 0 or negative dimension sized does not make sense here we must check for there aren't any
    for (dim_t i = 0; i < ndim; i++) assert(shape[i] > 0);

    tensor_t *t = malloc(sizeof(tensor_t));
    assert(t != NULL);

    t->ndim = ndim;
    t->shape = malloc(ndim * sizeof(*t->shape));
    assert(t->shape != NULL);
    memcpy(t->shape, shape, ndim * sizeof(*t->shape));

    t->stride = malloc(ndim * sizeof(*t->stride));
    assert(t->stride != NULL);
    t->stride[ndim-1] = 1;
    for (dim_t i = ndim-2; i >= 0; i--) t->stride[i] = t->stride[i+1] * t->shape[i+1];

    t->nelem = 0;
    for (dim_t i = 0; i < ndim; i++) t->nelem = (t->nelem > 0 ? t->nelem : 1) * t->shape[i];

    t->data = malloc(t->nelem * sizeof(*t->data));
    assert(t->data != NULL);

    return t;
}

/**
 * Frees all of the memory allocated to the tensor and sets its internal pointers to NULL.
 * The memory for the tensor_t struct is freed but it is not responsible for setting any variables pointing to it to NULL.
 * 
 * @param t tensor to free
 */
void tensor_free(tensor_t *t) {
    if (t == NULL) return;
    if (t->shape != NULL) {
        free(t->shape);
        t->shape = NULL;
    }
    if (t->stride != NULL) {
        free(t->stride);
        t->stride = NULL;
    }
    if (t->data != NULL) {
        free(t->data);
        t->data = NULL;
    }
    free(t);
}

/**
 * Transposes specified dimensions (interchanges shape and strides based on the specified dimensions; does not perform any copy)
 * 
 * @param t tensor to transpose
 * @param dim1 first dimension to transpose
 * @param dim2 second dimension to transpose
 * @return transposed tensor
 */
tensor_t *transpose(tensor_t *t, dim_t dim1, dim_t dim2) {
    assert(t != NULL);
    dim1 = RESOLVE_DIM(dim1, t->ndim);
    assert(dim1 < t->ndim);
    dim2 = RESOLVE_DIM(dim2, t->ndim);
    assert(dim2 < t->ndim);
    if (dim1 == dim2) return t; // if both dims are the same there's nothing to do
    // swap both shape and stride for the specified dimensions
    SWAP(t->shape[dim1], t->shape[dim2]);
    SWAP(t->stride[dim1], t->stride[dim2]);
    return t;
}

/**
 * Checks if tensor is contiguous
 * 
 * @param t tensor to check if contiguous
 * @return true if contiguous, false otherwise
 */
bool is_contiguous(tensor_t *t) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->stride != NULL);
    dim_sz_t mul = 1;
    for (dim_t i = t->ndim-1; i >= 0; i--) {
        if (t->stride[i] != mul) return false;
        mul *= t->shape[i];
    }
    return true;
}

/**
 * Converts the tensor into a contiguous tensor (creates a new data buffer and copies data).
 * 
 * @param t tensor to convert to contiguous
 * @return contiguous tensor
 */
tensor_t *contiguous(tensor_t *t) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->stride != NULL);
    if (is_contiguous(t)) return t;

    dim_sz_t *index = malloc(t->ndim * sizeof(*index));
    assert(index != NULL);

    float *data = malloc(t->nelem * sizeof(*data));
    assert(data != NULL);

    size_t idx = 0;
    for (uint32_t i = 0; i < t->nelem; i++) {
        data[i] = t->data[idx];
        for (dim_t d = t->ndim-1; d >= 0; d--) {
            index[d]++;
            idx += t->stride[d];
            if (index[d] < t->shape[d]) break;
            index[d] = 0;
            idx -= t->stride[d] * t->shape[d];
        }
    }

    free(index);
    free(t->data);
    t->data = data;
    t->stride[t->ndim-1] = 1;
    for (dim_t d = t->ndim-2; d >= 0; d--) t->stride[d] = t->shape[d+1] * t->stride[d+1];

    return t;
}

// **** code below does not work/untested with strides

/**
 * Broadcasts tensor a with tensor b and stores the new shapes into ashape and bshape respectively,
 * following NumPy's broadcast rules: https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
 * 
 * @param a tensor one to broadcast
 * @param ashape new shape for tensor one (pass NULL if not interested in new shape for tensor one)
 * @param b tensor two to broadcast
 * @param bshape new shape for tensor two (pass NULL if not interested in new shape for tensor two)
 * @return number of dimensions of the broadcasted shapes, or 0 if shapes are not broadcastable
 */
uint8_t broadcast(tensor_t *a, dim_sz_t **ashape, tensor_t *b, dim_sz_t **bshape) {
    assert(a != NULL);
    assert(b != NULL);
    assert(a->shape != NULL);
    assert(b->shape != NULL);

    dim_t ndim = MAX(a->ndim, b->ndim);
    dim_t offa = ndim - a->ndim, offb = ndim - b->ndim;

    dim_sz_t *_ashape = malloc(ndim * sizeof(*a->shape));
    assert(_ashape != NULL);
    dim_sz_t *_bshape = malloc(ndim * sizeof(*b->shape));
    assert(_bshape != NULL);

    for (dim_t i = 0; i < ndim; i++) {
        dim_sz_t sa = i < offa ? 1 : a->shape[i - offa];
        dim_sz_t sb = i < offb ? 1 : b->shape[i - offb];
        if (sa != sb && sa != 1 && sb != 1) {
            free(_ashape);
            free(_bshape);
            return 0;
        }
        _ashape[i] = sa;
        _bshape[i] = sb;
    }

    if (ashape != NULL) *ashape = _ashape;
    else free(_ashape);
    if (bshape != NULL) *bshape = _bshape;
    else free(_bshape);

    return ndim;
}

/**
 * Change the shape of a tensor
 * 
 * @param t tensor to change the shape of
 * @param ndim number of dimensions of the new shape
 * @param shape new shape
 * @return pointer to the reshaped tensor
 */
tensor_t *reshape(tensor_t *t, dim_t ndim, dim_sz_t *shape) {
    assert(t != NULL);

    uint32_t nelem = ndim > 0 ? 1 : 0;
    for (dim_t i = 0; i < ndim; i++) nelem *= shape[i];
    assert(t->nelem == nelem);

    t->shape = realloc(t->shape, ndim * sizeof(*t->shape));
    assert(t->shape != NULL);
    memcpy(t->shape, shape, ndim * sizeof(*shape));

    // TODO: reshape has to take into account the current strides, not just calculate them again
    t->stride = realloc(t->stride, ndim * sizeof(*t->stride));
    assert(t->stride != NULL);
    t->stride[ndim-1] = 1;
    for (dim_t i = ndim-2; i >= 0; i--) t->stride[i] = t->stride[i+1] * t->shape[i+1];

    t->ndim = ndim;

    return t;
}

/**
 * Removes the specified dimension of size 1
 * 
 * @param t tensor to remove the dimension of size 1
 * @param dim dimension of size 1 to remove
 * @return squeezed tensor
 */
tensor_t *squeeze(tensor_t *t, dim_t dim) {
    assert(t != NULL);
    assert(t->shape != NULL);
    dim_t d = RESOLVE_DIM(dim, t->ndim);
    if (d >= t->ndim || t->ndim < 2) return t;
    if (t->shape[d] != 1) return t;
    for (dim_t i = d; i < t->ndim-1; i++) t->shape[i] = t->shape[i+1];
    t->ndim--;
    return t;
}

/**
 * Adds a dimension of size 1 at the specified dimension
 * 
 * @param t tensor to add the dimension of size 1
 * @param dim dimension of size 1 to add
 * @return unsqueezed tensor
 */
tensor_t *unsqueeze(tensor_t *t, dim_t dim) {
    assert(t != NULL);
    assert(t->shape != NULL);
    dim_t d = RESOLVE_DIM(dim, t->ndim);
    if (d >= t->ndim) return t;
    dim_t ndim = t->ndim + 1;
    dim_sz_t *shape = malloc(ndim * sizeof(*shape));
    assert(shape != NULL);
    for (dim_t i = 0; i < d; i++) shape[i] = t->shape[i];
    shape[d] = 1;
    for (dim_t i = d; i < t->ndim; i++) shape[i+1] = t->shape[i];
    t->ndim = ndim;
    free(t->shape);
    t->shape = shape;
    return t;
}

// TODO: add argmin/argmax and amin/amax
// TODO: min/max functions can be simplified with ops (they are the exact same except for one symbol; like ewop)

/**
 * Returns the minimum value in a tensor
 * 
 * @param t tensor to search minimum value on
 * @return new tensor with the minimum value of t as its single element
 */
tensor_t *min(tensor_t *t) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->data != NULL);
    assert(t->nelem >= 1);
    tensor_t *r = tensor_alloc(1, (dim_sz_t[]){1});
    float m = t->data[0];
    for (uint32_t i = 1; i < t->nelem; i++) if (t->data[i] < m) m = t->data[i];
    *r->data = m;
    return r;
}

/**
 * Returns the maximum value in a tensor
 * 
 * @param t tensor to search maximum value on
 * @return new tensor with the maximum value of t as its single element
 */
tensor_t *max(tensor_t *t) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->data != NULL);
    assert(t->nelem >= 1);
    tensor_t *r = tensor_alloc(1, (dim_sz_t[]){1});
    float m = t->data[0];
    for (uint32_t i = 1; i < t->nelem; i++) if (t->data[i] > m) m = t->data[i];
    *r->data = m;
    return r;
}

/**
 * Returns the sum of all the elements of a tensor
 * 
 * @param t tensor to sum
 * @return sum of all the elements in `t`
 */
tensor_t *sumall(tensor_t *t) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->data != NULL);
    tensor_t *r = tensor_alloc(1, (dim_sz_t[]){1});
    float acc = 0;
    for (uint32_t i = 0; i < t->nelem; i++) acc += t->data[i];
    *r->data = acc;
    return r;
}

/**
 * Returns the sum of all the elements along the specified dimension of a tensor
 * 
 * @param t tensor to sum
 * @param dim dimension to sum along
 * @param keepdim  true to keep the summed dimension with a 1, false to squeeze the summed dimension
 * @return tensor with the summed elements along the specified dimension
 */
tensor_t *sum(tensor_t *t, int16_t dim, bool keepdim) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->data != NULL);
    dim_t d = RESOLVE_DIM(dim, t->ndim);
    assert(d < t->ndim);

    dim_sz_t *shape = malloc(t->ndim * sizeof(*shape));
    assert(shape != NULL);
    memcpy(shape, t->shape, t->ndim * sizeof(*shape));
    shape[d] = 1;
    tensor_t *r = tensor_alloc(t->ndim, shape);
    free(shape);

    // TODO: write about new reduce implementation in sum.ipynb
    uint32_t outer = 1, dimsz = t->shape[d], inner = 1;
    for (dim_t i = 0; i < d; i++) outer *= t->shape[i];
    for (dim_t i = d+1; i < t->ndim; i++) inner *= t->shape[i];

    // printf("shape: ");
    // print_shape(t->ndim, t->shape);
    // printf("\n");
    // printf("  dim: %d\n", d);
    // printf("outer: %d\n", outer);
    // printf("dimsz: %d\n", dimsz);
    // printf("inner: %d\n", inner);

    for (uint32_t o = 0; o < outer; o++) {
        for (uint32_t i = 0; i < inner; i++) {
            float acc = 0;
            // printf("r[%d] = ", o * inner + i);
            for (uint32_t j = 0; j < dimsz; j++) {
                acc += t->data[o * dimsz * inner + j * inner + i];
                // printf("t[%d]", o * dimsz * inner + j * inner + i);
                // if (j < dimsz-1) printf(" + ");
            }
            // printf("\n");
            r->data[o * inner + i] = acc;
        }
    }

    return keepdim ? r : squeeze(r, d);
}

// element wise operation
static tensor_t *ewop(tensor_t *a, tensor_t *b, tensor_op_t op) {
    assert(a != NULL);
    assert(b != NULL);

    dim_sz_t *ashape, *bshape;
    dim_t ndim = broadcast(a, &ashape, b, &bshape);
    assert(ndim > 0);

    dim_sz_t *cshape = malloc(ndim * sizeof(*cshape));
    assert(cshape != NULL);
    for (dim_t i = 0; i < ndim; i++) cshape[i] = MAX(ashape[i], bshape[i]);
    tensor_t *c = tensor_alloc(ndim, cshape);
    free(ashape);
    free(bshape);
    free(cshape);

    for (uint32_t cidx = 0; cidx < c->nelem; cidx++) {
        float aval = a->data[cidx % a->nelem];
        float bval = b->data[cidx % b->nelem];
        switch (op) {
            case OP_ADD: {
                c->data[cidx] = aval + bval;
                break;
            }
            case OP_MUL: {
                c->data[cidx] = aval * bval;
                break;
            }
            default: {
                break;
            }
        }
    }

    return c;
}

tensor_t *add(tensor_t *a, tensor_t *b) {
    return ewop(a, b, OP_ADD);
}

tensor_t *mul(tensor_t *a, tensor_t *b) {
    return ewop(a, b, OP_MUL);
}

/************************* HELPER FUNCTIONS *************************/

static uint8_t int_digits(double a) {
    uint8_t n = (uint8_t) fabs(a);
    if (n == 0) return 1;
    return (uint8_t) log10(n) + 1;
}

static bool has_decimals(double x) {
    double intpart;
    double frac = modf(x, &intpart);
    return frac != 0.0;
}

void tfprint(FILE *stream, tensor_t *t) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->ndim > 0);
    assert(t->stride != NULL);
    assert(t->data != NULL);

    // find out how many digits we need to print each element
    tensor_t *maxt = max(t);
    float maxel = *maxt->data;
    tensor_free(maxt);
    uint8_t ndigits = int_digits(maxel);
    bool decimals = false;
    for (uint32_t i = 0; i < t->nelem && !decimals; i++) decimals = has_decimals(t->data[i]);
    char *fmt = "%*g";
    if (decimals) {
        fmt = "%*.4f";
        ndigits += 5;
    }

    dim_t *index = malloc(t->ndim * sizeof(*index));
    assert(index != NULL);

    dim_t nnln = 0; // number of new lines to print closing
    uint32_t idx = 0; // index of current element in t->data
    for (uint32_t i = 0; i < t->nelem; i++) {
        if (nnln > 0) {
            for (dim_t j = 0; j < nnln; j++) fprintf(stream, "\n");
            nnln = 0;
        }

        // print necessary opening [; only performed when finished printing each row (every t->shape[t->ndim-1] elements)
        if (index[t->ndim-1] == 0) {
            dim_t nopen = 0; // number of opening [ to print
            for (dim_t d = t->ndim-1; d >= 0 && index[d] == 0; d--) nopen++;
            for (dim_t d = 0; d < t->ndim-nopen; d++) fprintf(stream, " ");
            for (dim_t d = t->ndim-nopen; d < t->ndim; d++) fprintf(stream, "[");
        }

        fprintf(stream, fmt, ndigits, t->data[idx]);
        if (index[t->ndim-1] < t->shape[t->ndim-1]-1) fprintf(stream, " ");

        for (dim_t d = t->ndim-1; d >= 0; d--) {
            index[d]++;
            idx += t->stride[d];
            if (index[d] < t->shape[d]) break;
            index[d] = 0;
            idx -= t->stride[d] * t->shape[d];
            fprintf(stream, "]");
            nnln++; // for each dimension, a new line should be printed, but only after all closing ]
        }
    }
    printf("\n");

    free(index);
}

void tprint(tensor_t *t) {
    tfprint(stdout, t);
}

void tprint_shape(uint32_t n, dim_sz_t *shape) {
    tfprint_shape(stdout, n, shape);
}

void tfprint_shape(FILE *stream, uint32_t n, dim_sz_t *shape) {
    if (shape == NULL) return;
    fprintf(stream, "(");
    for (uint8_t i = 0; i < n; i++) {
        fprintf(stream, "%d", shape[i]);
        if (i < n - 1) fprintf(stream, ", ");
    }
    fprintf(stream, ")");
}

void tprint_stride(uint32_t n, stride_t *stride) {
    tfprint_stride(stdout, n, stride);
}

void tfprint_stride(FILE *stream, uint32_t n, stride_t *stride) {
    if (stride == NULL) return;
    fprintf(stream, "(");
    for (uint8_t i = 0; i < n; i++) {
        fprintf(stream, "%u", stride[i]);
        if (i < n - 1) fprintf(stream, ", ");
    }
    fprintf(stream, ")");
}