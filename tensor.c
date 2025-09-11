#include "tensor.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define RESOLVE_DIM(dim, ndim) ((dim) >= 0 ? (dim) : (dim) + (ndim))

/**
 * Creates a tensor and allocates the required memory for it
 * 
 * @param ndim number of dimensions for the tensor (number of elements in the shape argument)
 * @param shape shape of the tensor
 * @return pointer to the created tensor
 */
tensor_t *tensor_alloc(uint8_t ndim, uint16_t *shape) {
    assert(shape != NULL);

    tensor_t *t = (tensor_t *) malloc(sizeof(tensor_t));
    assert(t != NULL);

    t->ndim = ndim;
    t->shape = (uint16_t *) malloc(ndim * sizeof(*t->shape));
    assert(t->shape != NULL);
    memcpy(t->shape, shape, ndim * sizeof(*t->shape));

    t->nelem = 0;
    for (uint8_t i = 0; i < ndim; i++) t->nelem = (t->nelem > 0 ? t->nelem : 1) * t->shape[i];

    t->data = (float *) malloc(t->nelem * sizeof(*t->data));
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
uint8_t broadcast(tensor_t *a, uint16_t **ashape, tensor_t *b, uint16_t **bshape) {
    assert(a != NULL);
    assert(b != NULL);
    assert(a->shape != NULL);
    assert(b->shape != NULL);

    uint8_t ndim = MAX(a->ndim, b->ndim);
    uint8_t offa = ndim - a->ndim, offb = ndim - b->ndim;

    uint16_t *_ashape = (uint16_t *) malloc(ndim * sizeof(*a->shape));
    assert(_ashape != NULL);
    uint16_t *_bshape = (uint16_t *) malloc(ndim * sizeof(*b->shape));
    assert(_bshape != NULL);

    for (uint8_t i = 0; i < ndim; i++) {
        uint16_t sa = i < offa ? 1 : a->shape[i - offa];
        uint16_t sb = i < offb ? 1 : b->shape[i - offb];
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
tensor_t *squeeze(tensor_t *t, int16_t dim) {
    assert(t != NULL);
    assert(t->shape != NULL);
    uint8_t d = RESOLVE_DIM(dim, t->ndim);
    if (d >= t->ndim || t->ndim < 2) return t;
    if (t->shape[d] != 1) return t;
    for (uint8_t i = d; i < t->ndim-1; i++) t->shape[i] = t->shape[i+1];
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
tensor_t *unsqueeze(tensor_t *t, int16_t dim) {
    assert(t != NULL);
    assert(t->shape != NULL);
    uint8_t d = RESOLVE_DIM(dim, t->ndim);
    if (d >= t->ndim) return t;
    uint8_t ndim = t->ndim + 1;
    uint16_t *shape = (uint16_t *) malloc(ndim * sizeof(*shape));
    assert(shape != NULL);
    for (uint8_t i = 0; i < d; i++) shape[i] = t->shape[i];
    shape[d] = 1;
    for (uint8_t i = d; i < t->ndim; i++) shape[i+1] = t->shape[i];
    t->ndim = ndim;
    free(t->shape);
    t->shape = shape;
    return t;
}

// TODO: transpose implementation
tensor_t *transpose(tensor_t *t, uint8_t dim1, uint8_t dim2) {
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
tensor_t *reshape(tensor_t *t, uint8_t ndim, uint16_t *shape) {
    assert(t != NULL);
    uint32_t nelem = 0;
    for (uint8_t i = 0; i < ndim; i++) nelem = (nelem > 0 ? nelem : 1) * shape[i];
    assert(t->nelem == nelem);
    uint16_t *new_shape = (uint16_t *) realloc(t->shape, ndim * sizeof(*t->shape));
    assert(new_shape != NULL);
    memcpy(new_shape, shape, ndim * sizeof(*shape));
    t->shape = new_shape;
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
    tensor_t *r = tensor_alloc(1, (uint16_t[]){1});
    assert(r != NULL);
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
    tensor_t *r = tensor_alloc(1, (uint16_t[]){1});
    assert(r != NULL);
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
    tensor_t *r = tensor_alloc(1, (uint16_t[]){1});
    if (r == NULL) return NULL;
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
    uint8_t d = RESOLVE_DIM(dim, t->ndim);
    assert(d < t->ndim);

    uint16_t *shape = (uint16_t *) malloc(t->ndim * sizeof(*shape));
    assert(shape != NULL);
    for (uint8_t i = 0; i < t->ndim; i++) shape[i] = t->shape[i];
    // memcpy(shape, t->shape, t->ndim * sizeof(*shape));
    shape[d] = 1;
    tensor_t *r = tensor_alloc(t->ndim, shape);
    assert(r != NULL);
    free(shape);

    // TODO: write about new reduce implementation in sum.ipynb
    uint32_t outer = 1, dimsz = t->shape[d], inner = 1;
    for (uint8_t i = 0; i < d; i++) outer *= t->shape[i];
    for (uint8_t i = d+1; i < t->ndim; i++) inner *= t->shape[i];

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

    uint16_t *ashape, *bshape;
    uint8_t ndim = broadcast(a, &ashape, b, &bshape);
    assert(ndim > 0);

    uint16_t *cshape = (uint16_t *) malloc(ndim * sizeof(*cshape));
    assert(cshape != NULL);
    uint32_t nelem = ndim > 0 ? 1 : 0;
    for (uint8_t i = 0; i < ndim; i++) {
        cshape[i] = MAX(ashape[i], bshape[i]);
        nelem *= cshape[i];
    }
    tensor_t *c = tensor_alloc(ndim, cshape);
    assert(c != NULL);
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

// tensor_t *dot(tensor_t *a, tensor_t *b) {
//     if (a == NULL || b == NULL || (a->ndim < 2 && b->ndim < 2)) return NULL;

//     printf("a.shape: ");
//     shape_print(stdout, a->ndim, a->shape);
//     printf("\n");

//     unsqueeze(b, b->ndim-2);
//     printf("b.shape: ");
//     shape_print(stdout, b->ndim, b->shape);
//     printf("\n");

//     uint16_t *ashape, *bshape;
//     b->ndim--; // last dim of a must match second to last dim of b
//     uint8_t ndim = broadcast(a, &ashape, b, &bshape);
//     b->ndim++;
//     printf("ndim: %d\n", ndim);
//     if (ndim == 0) return NULL;

//     printf("ashape: ");
//     shape_print(stdout, ndim, ashape);
//     printf("\n");
//     printf("bshape: ");
//     shape_print(stdout, ndim, bshape);
//     printf("\n");

//     uint16_t *cshape = (uint16_t *) malloc(ndim * sizeof(*cshape));
//     if (cshape == NULL) return NULL;
//     uint32_t nelem = 0;
//     for (uint8_t i = 0; i < ndim; i++) {
//         cshape[i] = MAX(ashape[i], bshape[i]);
//         nelem = (nelem > 0 ? nelem : 1) * cshape[i];
//     }
//     cshape[ndim-1] = b->shape[b->ndim-1];
//     tensor_t *c = tensor_alloc(ndim, cshape);
//     if (c == NULL) {
//         free(ashape);
//         free(bshape);
//         free(cshape);
//         return NULL;
//     }
//     printf("cshape: ");
//     shape_print(stdout, ndim, cshape);
//     printf("\n");

//     printf("a:\n");
//     print(a);
//     printf("\n");

//     printf("b:\n");
//     print(b);
//     printf("\n");

//     uint32_t aoff = 0, boff = 0, coff = 0;
//     uint32_t astep = ashape[a->ndim-1] * ashape[a->ndim-2];
//     uint32_t bstep = bshape[b->ndim-1] * bshape[b->ndim-2];
//     uint32_t cstep = cshape[c->ndim-1] * cshape[c->ndim-2];
//     uint32_t nitr = c->nelem / cstep;

//     // coff += cstep;
//     nitr = 1;

//     /**
//      * a.shape = (2, 2, 3)
//      * b.shape = (3, 2)
//      * c.shape = (2, 2, 2)
//      * 
//      * c[0,0,0] = a[0,0] * b[:,0]
//      * 
//      * c[0,0,0] = a[0,0,0]*b[0,0] + a[0,0,1]*b[1,0] + a[0,0,2]*b[2,0]
//      * c[0,0,1] = a[0,0,0]*b[0,1] + a[0,0,1]*b[1,1] + a[0,0,2]*b[2,1]
//      * c[0,1,0] = a[0,1,0]*b[0,0] + a[0,1,1]*b[1,0] + a[0,1,2]*b[2,0]
//      * c[0,1,1] = a[0,1,0]*b[0,1] + a[0,1,1]*b[1,1] + a[0,1,2]*b[2,1]
//      * c[1,0,0] = a[1,0,0]*b[0,0] + a[1,0,1]*b[1,0] + a[1,0,2]*b[2,0]
//      * c[1,0,1] = a[1,0,0]*b[0,1] + a[1,0,1]*b[1,1] + a[1,0,2]*b[2,1]
//      * c[1,1,0] = a[1,1,0]*b[0,0] + a[1,1,1]*b[1,0] + a[1,1,2]*b[2,0]
//      * c[1,1,1] = a[1,1,0]*b[0,1] + a[1,1,1]*b[1,1] + a[1,1,2]*b[2,1]
//     */

//     uint8_t din = ashape[a->ndim-2], dmid = bshape[b->ndim-2], dout = cshape[c->ndim-1];
//     for (uint16_t itr = 0; itr < nitr; itr++) {
//         for (uint8_t i = 0; i < din; i++) {
//             for (uint8_t o = 0; o < dout; o++) {

//                 float acc = 0;
//                 printf("c[%d] = ", i * dout + o + coff);
//                 for (uint8_t m = 0; m < dmid; m++) {
//                     printf("%.4f * %.4f", a->data[i * dmid + m + aoff], b->data[m * dout + o + boff]);
//                     acc += a->data[i * dmid + m + aoff] * b->data[m * dout + o + boff];
//                     if (m < dmid - 1) printf(" + ");
//                 }
//                 printf("\n");
//                 c->data[i * dout + o + coff] = acc;

//             }
//             // coff += cstep;
//         }
//         // coff += cstep;
//         // boff = (boff + bstep) % b->nelem;
//         // aoff = (aoff + astep) % a->nelem;
//     }

//     printf("c:\n");
//     print(c);
//     printf("\n");

//     free(ashape);
//     free(bshape);
//     free(cshape);

//     return NULL;
// }

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

void fprint(FILE *stream, tensor_t *t) {
    if (t == NULL || t->shape == NULL || t->data == NULL) return;

    // shape multiples (how many elements fit in each dimension)
    uint16_t *mul = (uint16_t *) malloc(t->ndim * sizeof(*t->shape));
    if (mul == NULL) return;
    uint8_t ndim = t->ndim;
    mul[ndim-1] = t->shape[ndim-1];
    for (int8_t i = ndim-2; i >= 0; i--) mul[i] = mul[i+1] * t->shape[i];

    // shape modulus for current index (shape[i] = idx % shape[i])
    // no need to save the last dimension's modulus as the loop steps by its value (will always be 0)
    uint16_t *mod = (uint16_t *) malloc((t->ndim - 1) * sizeof(*t->shape));
    if (mod == NULL) {
        free(mul);
        return;
    }

    tensor_t *maxt = max(t);
    if (maxt == NULL) return;
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

    for (uint32_t i = 0; i < t->nelem;) {
        for (uint8_t dim = 0; dim < ndim-1; dim++) {
            mod[dim] = i % mul[dim];
            if (mod[dim] == 0) {
                for (uint8_t j = 0; j < ndim-dim-1 && i > 0; j++) fprintf(stream, "\n");
                break;
            }
        }

        for (uint8_t j = 0; j < ndim-1; j++) fprintf(stream, mod[j] == 0 ? "[" : " ");
        fprintf(stream, "[");
        for (uint16_t j = 0; j < mul[ndim-1]; j++) {
            fprintf(stream, fmt, ndigits, t->data[i++]);
            if (j < mul[ndim-1] - 1) fprintf(stream, " ");
        }
        for (uint8_t j = 0; j < ndim-1; j++) if (mod[j] == mul[j] - mul[ndim-1]) fprintf(stream, "]");
        fprintf(stream, "]\n");
    }

    free(mul);
    free(mod);
}

void print(tensor_t *t) {
    fprint(stdout, t);
}

void print_shape(uint8_t ndim, uint16_t *shape) {
    fprint_shape(stdout, ndim, shape);
}

void fprint_shape(FILE *stream, uint8_t ndim, uint16_t *shape) {
    if (shape == NULL) return;
    fprintf(stream, "(");
    for (uint8_t i = 0; i < ndim; i++) {
        fprintf(stream, "%d", shape[i]);
        if (i < ndim - 1) fprintf(stream, ", ");
    }
    fprintf(stream, ")");
}
