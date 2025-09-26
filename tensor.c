#include "tensor.h"
#include "debug.h"
#include "color.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define SWAP(a, b) do { typeof(a) tmp = (a); (a) = (b); (b) = tmp; } while (0)

#define BUFF_SIZE 512
static char buff[BUFF_SIZE];

static size_t tinfo2str(tensor_t *t, char *dst, const size_t dstlen);
static size_t tuple2str(void *tuple, const size_t numel, const size_t szel, const char *const fmtel, char *dst, const size_t dstlen);

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

    t->numel = 0;
    for (dim_t i = 0; i < ndim; i++) t->numel = (t->numel > 0 ? t->numel : 1) * t->shape[i];

    t->data = malloc(t->numel * sizeof(*t->data));
    assert(t->data != NULL);

    DBG(1, {
        assert(tinfo2str(t, buff, BUFF_SIZE) != 0);
        uint64_t sz = sizeof(tensor_t) + t->ndim * (sizeof(*t->shape) + sizeof(*t->stride)) + t->numel * sizeof(*t->data);
        printf("%s numel=%u sz=%llu", buff, t->numel, sz);
    });

    return t;
}

/**
 * Frees all of the memory allocated to the tensor and sets its internal pointers to NULL.
 * The memory for the tensor_t struct is freed but it is not responsible for setting any variables pointing to it to NULL.
 * 
 * @param t tensor to free
 */
void tensor_free(tensor_t *t) {
    if (t != NULL) {
        DBG(1, {
            assert(tinfo2str(t, buff, BUFF_SIZE) != 0);
            uint64_t sz = sizeof(tensor_t) + t->ndim * (sizeof(*t->shape) + sizeof(*t->stride)) + t->numel * sizeof(*t->data);
            printf("%s numel=%u sz=%llu", buff, t->numel, sz);
        });
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
}

/**
 * Creates a tensor with the range [start, end) and the specified step size.
 * The number of elements in the tensor will be: (end - start) / step
 * 
 * @param start start value (first element)
 * @param end stop value (value after the last element)
 * @param step step value (difference between any two contiguous elements)
 * @return tensor filled with the specified range
 */
tensor_t *range(float start, float end, float step) {
    assert(start < end);
    uint32_t numel = (end - start) / step;
    tensor_t *t = tensor_alloc(1, (dim_sz_t[]){numel});
    float acc = start;
    for (uint32_t i = 0; i < t->numel; i++) {
        t->data[i] = acc;
        acc += step;
    }
    return t;
}

/**
 * Creates a tensor with numel elements and sets all of the to value.
 * 
 * @param numel number of elements in the tensor
 * @param value value to set all elements to
 * @return filled tensor
 */
tensor_t *fill(uint32_t numel, float value) {
    assert(numel > 0);
    tensor_t *t = tensor_alloc(1, (dim_sz_t[]){numel});
    memset(t->data, value, t->numel * sizeof(value));
    return t;
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
    dim1 = resolve_dim(t->ndim, dim1);
    dim2 = resolve_dim(t->ndim, dim2);
    DBG(1, {
        assert(tinfo2str(t, buff, BUFF_SIZE) > 0);
        printf("%s - [%d, %d]", buff, dim1, dim2);
    });
    if (dim1 != dim2) {
        // swap both shape and stride for the specified dimensions
        SWAP(t->shape[dim1], t->shape[dim2]);
        SWAP(t->stride[dim1], t->stride[dim2]);
    }
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
    bool ret = true;
    dim_sz_t mul = 1;
    for (dim_t i = t->ndim-1; i >= 0 && ret; i--) {
        if (t->stride[i] != mul) ret = false;
        else mul *= t->shape[i];
    }
    DBG(2, {
        assert(tinfo2str(t, buff, BUFF_SIZE) > 0);
        printf("%s %s", buff, ret ? GRN "true" : RED "false");
    });
    return ret;
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
    bool c = is_contiguous(t);
    DBG(1, {
        assert(tinfo2str(t, buff, BUFF_SIZE) != 0);
        printf("%s %s", buff, c ? GRN "nocopy" RST : RED "copy" RST);
    });
    if (!c) {
        dim_sz_t *index = malloc(t->ndim * sizeof(*index));
        assert(index != NULL);
        memset(index, 0, t->ndim * sizeof(*index));

        float *data = malloc(t->numel * sizeof(*data));
        assert(data != NULL);

        size_t idx = 0;
        for (uint32_t i = 0; i < t->numel; i++) {
            data[i] = t->data[idx];
            // increase the index for the last dimension and carry the overflow to the previous ones
            for (dim_t d = t->ndim-1; d >= 0; d--) {
                index[d]++;
                idx += t->stride[d]; // increase idx by the current dimension's stride
                if (index[d] < t->shape[d]) break;
                index[d] = 0;
                idx -= t->stride[d] * t->shape[d]; // rollback idx to the previous dimension's base
            }
        }

        free(index);
        free(t->data);
        t->data = data;
        // recalculate strides and tensor is now contiguous
        t->stride[t->ndim-1] = 1;
        for (dim_t d = t->ndim-2; d >= 0; d--) t->stride[d] = t->shape[d+1] * t->stride[d+1];
    }

    return t;
}

/**
 * Resolves shapes with a negative element.
 * 
 * @param numel number of total elements in tensor
 * @param ndim number of dimensions in shape
 * @param shape shape to resolve
 * @return dimension that was resolved, -1 if shape had no negative element
 */
dim_t resolve_shape(uint32_t numel, dim_t ndim, dim_sz_t *shape) {
    assert(ndim > 0);
    assert(shape != NULL);

    dim_t dim = -1;
    uint32_t mul = 1;
    for (dim_t d = 0; d < ndim; d++) {
        if (shape[d] < 0) {
            assert(dim < 0);
            dim = d;
        } else {
            mul *= shape[d];
        }
    }

    int8_t lvl = 3;
    dim_sz_t *prev = NULL;
    if (dbglvl() >= lvl) {
        prev = malloc(ndim * sizeof(*prev));
        assert(prev);
        memcpy(prev, shape, ndim * sizeof(*prev));
    }

    if (dim > 0) shape[dim] = numel / mul;

    DBG(lvl, {
        assert(tuple2str(prev, ndim, sizeof(*prev), "%d", buff, BUFF_SIZE) > 0);
        printf("%s -> ", buff);
        assert(tuple2str(shape, ndim, sizeof(*shape), "%d", buff, BUFF_SIZE) > 0);
        printf("%s", buff);
        free(prev);
    });

    return dim;
}

/**
 * Resolves the dimension with possible negative elements.
 * Also asserts that resulting dim is within [0, ndim-1]
 * 
 * @param ndim number of dimensions
 * @param dim dimension to resolve
 * @return resolved dimension
 */
dim_t resolve_dim(dim_t ndim, dim_t dim) {
    dim_t d = dim >= 0 ? dim : dim + ndim;
    DBG(3, { printf("%d [0..%d] -> %d", dim, ndim, d); });
    assert(d < ndim);
    return d;
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

    resolve_shape(t->numel, ndim, shape);
    uint32_t numel = ndim > 0 ? 1 : 0;
    for (dim_t i = 0; i < ndim; i++) numel *= shape[i];
    assert(t->numel == numel);

    // TODO: for certain non-contiguous cases, the strides can be computed without needing a contiguous copy
    //       according to ChatGPT: https://chatgpt.com/share/68d57068-038c-8004-bb0b-56be52f7577a
    //       keeping it simple for now but definitely interesting to explore
    bool c = is_contiguous(t);
    DBG(1, {
        assert(tinfo2str(t, buff, BUFF_SIZE) > 0);
        printf("%s %s", buff, c ? GRN "stride only" RST : RED "copy" RST);
    });
    if (!c) contiguous(t);

    t->stride = realloc(t->stride, ndim * sizeof(*t->stride));
    assert(t->stride != NULL);
    t->stride[ndim-1] = 1;
    for (dim_t i = ndim-2; i >= 0; i--) t->stride[i] = t->stride[i+1] * shape[i+1];

    t->ndim = ndim;
    t->shape = realloc(t->shape, t->ndim * sizeof(*t->shape));
    assert(t->shape != NULL);
    memcpy(t->shape, shape, ndim * sizeof(*shape));

    return t;
}

/**
 * Broadcasts shape a with shape b and stores the new shapes into ashape and bshape respectively,
 * following NumPy's broadcast rules: https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
 * The resulting adst and bdst shapes will be the same as their asrc and bsrc shapes with filled in 1's
 * where necessary to make them broadcastable.
 * 
 * @param andim number of dimensions of shape a
 * @param asrc shape a
 * @param adst new shape a (pass NULL if not interested in new shape)
 * @param bndim number of dimensions of shape b
 * @param bsrc shape b
 * @param bdst new shape b (pass NULL if not interested in new shape)
 * @return number of dimensions of the broadcasted shapes, or 0 if shapes are not broadcastable
 */
uint8_t broadcast(dim_t andim, dim_sz_t *asrc, dim_sz_t **adst, dim_t bndim, dim_sz_t *bsrc, dim_sz_t **bdst) {
    assert(asrc != NULL);
    assert(bsrc != NULL);

    dim_t ndim = MAX(andim, bndim);
    dim_t offa = ndim - andim, offb = ndim - bndim;

    dim_sz_t *ashape = malloc(ndim * sizeof(*asrc));
    assert(ashape != NULL);
    dim_sz_t *bshape = malloc(ndim * sizeof(*bsrc));
    assert(bshape != NULL);

    // align each shape src to the right and check if each of their size match (or is 1)
    for (dim_t i = 0; i < ndim; i++) {
        dim_sz_t sa = i < offa ? 1 : asrc[i - offa];
        dim_sz_t sb = i < offb ? 1 : bsrc[i - offb];
        if (sa != sb && sa != 1 && sb != 1) {
            free(ashape);
            free(bshape);
            return 0;
        }
        ashape[i] = sa;
        bshape[i] = sb;
    }

    if (adst != NULL) *adst = ashape;
    else free(ashape);
    if (bdst != NULL) *bdst = bshape;
    else free(bshape);

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

    dim_t d = resolve_dim(t->ndim, dim);
    if (t->shape[d] != 1 || t->ndim == 1) return t;
    for (dim_t i = d; i < t->ndim-1; i++) {
        t->shape[i] = t->shape[i+1];
        t->stride[i] = t->stride[i+1];
    }
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
    assert(t->stride != NULL);

    dim_t ndim = t->ndim + 1;
    dim_t d = resolve_dim(ndim, dim);

    dim_sz_t *shape = malloc(ndim * sizeof(*shape));
    assert(shape != NULL);
    memcpy(shape, t->shape, t->ndim * sizeof(*shape));

    stride_t *stride = malloc(ndim * sizeof(*stride));
    assert(stride != NULL);
    memcpy(stride, t->stride, t->ndim * sizeof(*stride));

    for (dim_t i = t->ndim; i > d; i--) {
        shape[i] = shape[i-1];
        stride[i] = stride[i-1];
    }
    shape[d] = 1;
    stride[d] = d+1 < ndim ? shape[d+1] * stride[d+1] : 1;

    free(t->shape);
    t->shape = shape;
    free(t->stride);
    t->stride = stride;
    t->ndim = ndim;

    return t;
}

// **** code below does not work/untested with strides

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
    assert(t->numel >= 1);
    tensor_t *r = tensor_alloc(1, (dim_sz_t[]){1});
    float m = t->data[0];
    for (uint32_t i = 1; i < t->numel; i++) if (t->data[i] < m) m = t->data[i];
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
    assert(t->numel >= 1);
    tensor_t *r = tensor_alloc(1, (dim_sz_t[]){1});
    float m = t->data[0];
    for (uint32_t i = 1; i < t->numel; i++) if (t->data[i] > m) m = t->data[i];
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
    for (uint32_t i = 0; i < t->numel; i++) acc += t->data[i];
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
tensor_t *sum(tensor_t *t, dim_t dim, bool keepdim) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->data != NULL);

    dim_t d = resolve_dim(t->ndim, dim);
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
    dim_t ndim = broadcast(a->ndim, a->shape, &ashape, b->ndim, b->shape, &bshape);
    assert(ndim > 0);

    dim_sz_t *cshape = malloc(ndim * sizeof(*cshape));
    assert(cshape != NULL);
    for (dim_t i = 0; i < ndim; i++) cshape[i] = MAX(ashape[i], bshape[i]);
    tensor_t *c = tensor_alloc(ndim, cshape);
    free(ashape);
    free(bshape);
    free(cshape);

    for (uint32_t cidx = 0; cidx < c->numel; cidx++) {
        float aval = a->data[cidx % a->numel];
        float bval = b->data[cidx % b->numel];
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

static size_t tuple2str(void *tuple, const size_t numel, const size_t szel, const char *const fmtel, char *dst, const size_t dstlen) {
    int w = 0; // number of written characters by snprintf
    size_t o = 0; // dst current offset and length

    w = snprintf(&dst[o], dstlen-o, "(");
    if (w < 0 || o+w+1 >= dstlen) return 0;
    o += w;

    for (size_t i = 0; i < numel; i++) {
        if (szel == 0 || tuple == NULL) {
            w = snprintf(&dst[o], dstlen-o, "%s", fmtel);
        } else if (szel == sizeof(uint8_t)) {
            w = snprintf(&dst[o], dstlen-o, fmtel, *(uint8_t*)(tuple + i*szel));
        } else if (szel == sizeof(uint16_t)) {
            w = snprintf(&dst[o], dstlen-o, fmtel, *(uint16_t*)(tuple + i*szel));
        } else if (szel == sizeof(uint32_t)) {
            w = snprintf(&dst[o], dstlen-o, fmtel, *(uint32_t*)(tuple + i*szel));
        } else if (szel == sizeof(uint64_t)) {
            w = snprintf(&dst[o], dstlen-o, fmtel, *(uint64_t*)(tuple + i*szel));
        } else {
            return 0;
        }
        if (w < 0 || o+w+1 >= dstlen) return 0;
        o += w;
        if (i < numel-1) {
            w = snprintf(&dst[o], dstlen-o, ", ");
            if (w < 0 || o+w+1 >= dstlen) return 0;
            o += w;
        }
    }

    w = snprintf(&dst[o], dstlen-o, ")");
    if (w < 0 || o+w+1 >= dstlen) return 0;
    o += w;

    return o;
}

static size_t tinfo2str(tensor_t *t, char *dst, const size_t dstlen) {
    int w = 0; // number of written characters by snprintf
    size_t o = 0; // dst current offset and length

    w = snprintf(&dst[o], dstlen-o, "t=%p shape=", t);
    if (w < 0 || o+w+1 >= dstlen) return 0;
    o += w;

    w = tuple2str(t->shape, t->ndim, sizeof(*t->shape), "%d", &dst[o], dstlen-o);
    if (w < 0 || o+w+1 >= dstlen) return 0;
    o += w;

    w = snprintf(&dst[o], dstlen-o, " stride=");
    if (w < 0 || o+w+1 >= dstlen) return 0;
    o += w;

    w = tuple2str(t->stride, t->ndim, sizeof(*t->stride), "%u", &dst[o], dstlen-o);
    if (w < 0 || o+w+1 >= dstlen) return 0;
    o += w;

    return o;
}

void tfinfo(FILE *stream, tensor_t *t) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->ndim > 0);
    assert(t->stride != NULL);
    assert(t->data != NULL);
    assert(tinfo2str(t, buff, BUFF_SIZE) > 0);
    fprintf(stream, "%s\n", buff);
}

void tinfo(tensor_t *t) {
    tfinfo(stdout, t);
}

void tfprint(FILE *stream, tensor_t *t) {
    assert(t != NULL);
    assert(t->shape != NULL);
    assert(t->ndim > 0);
    assert(t->stride != NULL);
    assert(t->data != NULL);

    // find out how many digits we need to print each element
    float maxel = t->data[0];
    for (uint32_t i = 1; i < t->numel; i++) if (t->data[i] > maxel) maxel = t->data[i];
    uint8_t ndigits = int_digits(maxel);
    bool decimals = false;
    for (uint32_t i = 0; i < t->numel && !decimals; i++) decimals = has_decimals(t->data[i]);
    char *fmt = "%*g";
    if (decimals) {
        fmt = "%*.4f";
        ndigits += 5;
    }

    dim_t *index = malloc(t->ndim * sizeof(*index));
    assert(index != NULL);
    memset(index, 0, t->ndim * sizeof(*index));

    dim_t nnln = 0; // number of new lines to print closing
    uint32_t idx = 0; // index of current element in t->data
    for (uint32_t i = 0; i < t->numel; i++) {
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
    if (shape == NULL) n = 0;
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
    if (stride == NULL) n = 0;
    fprintf(stream, "(");
    for (uint8_t i = 0; i < n; i++) {
        fprintf(stream, "%u", stride[i]);
        if (i < n - 1) fprintf(stream, ", ");
    }
    fprintf(stream, ")");
}