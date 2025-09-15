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

tensor_t *transpose(tensor_t *t, dim_t dim1, dim_t dim2) {
    dim1 = RESOLVE_DIM(dim1, t->ndim);
    assert(dim1 < t->ndim);
    dim2 = RESOLVE_DIM(dim2, t->ndim);
    assert(dim2 < t->ndim);
    if (dim1 == dim2) return t;
    SWAP(t->shape[dim1], t->shape[dim2]);
    SWAP(t->stride[dim1], t->stride[dim2]);
    return t;
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

    t->stride = realloc(t->stride, ndim * sizeof(*t->stride));
    assert(t->stride != NULL);
    t->stride[ndim-1] = 1;
    for (dim_t i = ndim-2; i >= 0; i--) t->stride[i] = t->stride[i+1] * t->shape[i+1];

    t->ndim = ndim;

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

static bool tfprint_data(FILE *stream, const char *fmt, uint8_t ndigits, tensor_t *t, dim_t dim, dim_sz_t *indeces, uint32_t *mul) {
    if (dim == t->ndim-1) {
        // calculate current position in the printed tensor: [0, t->nelem]
        uint32_t pos = 0;
        for (dim_t d = 0; d < dim; d++) pos += indeces[d] * mul[d];
        // if current pos is a multiple of dim size -> beginning of array
        for (dim_t d = 0; d < dim; d++) fprintf(stream, pos % (t->shape[d] * mul[d]) == 0 ? "[" : " ");

        // calculate current offset in t->data and print next row
        uint32_t off = 0;
        for (dim_t d = 0; d < dim; d++) off += indeces[d] * t->stride[d];
        fprintf(stream, "[");
        for (dim_sz_t i = 0; i < t->shape[dim]; i++) {
            fprintf(stream, fmt, ndigits, t->data[off + i * t->stride[dim]]);
            if (i < t->shape[dim]-1) printf(" ");
        }
        fprintf(stream, "]");

        pos += t->shape[dim]; // move pos to the end of what has just been printed
        // if current pos is a multiple of dim size -> end of array
        for (dim_t d = 0; d < dim; d++) if (pos % (t->shape[d] * mul[d]) == 0) printf("]");
        return pos == t->nelem; // let the caller know whether all the elements have been printed
    }

    bool end = false; // finished printing all elements?
    for (dim_sz_t i = 0; i < t->shape[dim]; i++) {
        indeces[dim] = i;
        if (tfprint_data(stream, fmt, ndigits, t, dim+1, indeces, mul)) end = true;
        if (!end) fprintf(stream, "\n");
    }
    if (end && dim == 0) fprintf(stream, "\n");
    indeces[dim] = 0;

    return end;
}

void tfprint(FILE *stream, tensor_t *t) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->ndim > 0);
    assert(t->stride != NULL);
    assert(t->data != NULL);

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

    dim_sz_t *indeces = malloc(t->ndim * sizeof(*indeces));
    assert(indeces != NULL);
    memset(indeces, 0, t->ndim * sizeof(*indeces));

    dim_sz_t *mul = malloc(t->ndim * sizeof(*mul));
    assert(mul != NULL);
    mul[t->ndim-1] = 1;
    for (dim_t i = t->ndim-2; i >= 0; i--) mul[i] = mul[i+1] * t->shape[i+1];

    tfprint_data(stream, fmt, ndigits, t, 0, indeces, mul);
    free(indeces);
    free(mul);
}

void tprint(tensor_t *t) {
    tfprint(stdout, t);
}

void tfprint_tuple(FILE *stream, uint32_t n, uint32_t *tuple) {
    if (tuple == NULL) return;
    fprintf(stream, "(");
    for (uint8_t i = 0; i < n; i++) {
        fprintf(stream, "%d", tuple[i]);
        if (i < n - 1) fprintf(stream, ", ");
    }
    fprintf(stream, ")");
}

void tprint_tuple(uint32_t n, uint32_t *tuple) {
    tfprint_tuple(stdout, n, tuple);
}