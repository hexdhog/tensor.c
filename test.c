#include "tensor.h"
#include <assert.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdbool.h>
#include <stdio.h>
#include <fcntl.h>

#define CHECK_ABORT(code) do { \
    pid_t pid = fork(); \
    assert(pid >= 0); \
    if (pid == 0) { \
        int fd = open("/dev/null", O_WRONLY); \
        dup2(fd, STDERR_FILENO); \
        close(fd); \
        do { code } while (0); \
        exit(1); \
    } else { \
        int status; \
        waitpid(pid, &status, 0); \
        assert(WIFSIGNALED(status)); \
        assert(WTERMSIG(status) == SIGABRT); \
    } \
} while (0)

/********************* TRANSPOSE *********************/

const char *test_transpose() {
    tensor_t *t = tensor_alloc(2, (dim_sz_t[]){2, 3});
    for (int i = 0; i < 6; i++) t->data[i] = i + 1; // [1,2,3,4,5,6]
    assert(memcmp(t->shape, (dim_sz_t[]){2, 3}, t->ndim * sizeof(*t->shape)) == 0);
    assert(memcmp(t->stride, (dim_sz_t[]){3, 1}, t->ndim * sizeof(*t->stride)) == 0);
    transpose(t, 0, 1);
    assert(memcmp(t->shape, (dim_sz_t[]){3, 2}, t->ndim * sizeof(*t->shape)) == 0);
    assert(memcmp(t->stride, (dim_sz_t[]){1, 3}, t->ndim * sizeof(*t->stride)) == 0);

    tensor_free(t);

    return __func__;
}

/********************* CONTIGUOUS *********************/

const char *test_is_contiguous() {
    tensor_t *t = tensor_alloc(2, (dim_sz_t[]){2, 3});
    for (int i = 0; i < 6; i++) t->data[i] = i + 1; // [1,2,3,4,5,6]
    assert(is_contiguous(t));
    transpose(t, 0, 1);
    assert(!is_contiguous(t));
    contiguous(t);
    assert(is_contiguous(t));
    assert(memcmp(t->stride, (stride_t[]){2, 1}, t->ndim * sizeof(*t->stride)) == 0);
    assert(memcmp(t->data, (float[]){1, 4, 2, 5, 3, 6}, t->numel * sizeof(*t->data)) == 0);

    tensor_free(t);

    return __func__;
}

/********************* RESHAPE *********************/

const char *test_reshape() {
    // contiguous, computes strides directy
    // {
    //     tensor_t *t = tensor_alloc(1, (dim_sz_t[]){6});
    //     reshape(t, 2, (dim_sz_t[]){2, 3});
    //     assert(t->ndim == 2);
    //     assert(memcmp(t->stride, (stride_t[]){3, 1}, t->ndim * sizeof(*t->stride)) == 0);
    //     tensor_free(t);
    // }

    // contiguous but compatible, tries to match strides
    // {
    //     tensor_t *t = tensor_alloc(2, (dim_sz_t[]){2, 3});
    //     for (uint32_t i = 0; i < t->numel; i++) t->data[i] = i + 1;
    //     transpose(t, 0, 1);
    //     reshape(t, 1, (dim_sz_t[]){6});
    //     assert(t->ndim == 1);
    //     assert(memcmp(t->stride, (stride_t[]){1}, t->ndim * sizeof(*t->stride)) == 0);
    //     tensor_free(t);
    // }

    // non-contiguous but not compatible, performs contiguous copy
    {
        tensor_t *t = tensor_alloc(2, (dim_sz_t[]){2, 3});
        for (uint32_t i = 0; i < t->numel; i++) t->data[i] = i + 1;
        transpose(t, 0, 1);
        assert(t->ndim == 2);
        assert(memcmp(t->shape, (stride_t[]){3, 2}, t->ndim * sizeof(*t->shape)) == 0);
        assert(memcmp(t->stride, (stride_t[]){1, 3}, t->ndim * sizeof(*t->stride)) == 0);
        reshape(t, 2, (dim_sz_t[]){2, 3});
        assert(t->ndim == 2);
        // assert(memcmp(t->shape, (stride_t[]){6}, t->ndim * sizeof(*t->shape)) == 0);
        // assert(memcmp(t->stride, (stride_t[]){1}, t->ndim * sizeof(*t->stride)) == 0);
        tensor_free(t);
    }

    return __func__;
}

/********************* SUM *********************/

const char *test_sumall() {
    tensor_t *t = tensor_alloc(2, (dim_sz_t[]){2, 3});
    for (int i = 0; i < 6; i++) t->data[i] = i + 1; // [1,2,3,4,5,6]

    tensor_t *r = sumall(t);
    assert(r != NULL);
    assert(r->ndim == 0 || (r->ndim == 1 && r->shape[0] == 1));
    assert(*r->data == 21.0f);

    tensor_free(t);
    tensor_free(r);

    return __func__;
}

const char *test_sum_dim0() {
    tensor_t *t = tensor_alloc(2, (dim_sz_t[]){2, 3});
    float vals[] = {1,2,3,4,5,6};
    for (int i = 0; i < 6; i++) t->data[i] = vals[i];

    tensor_t *r = sum(t, 0, false);
    assert(r != NULL);
    assert(r->ndim == 1 && r->shape[0] == 3);
    assert(r->data[0] == 1+4);
    assert(r->data[1] == 2+5);
    assert(r->data[2] == 3+6);

    tensor_free(t);
    tensor_free(r);

    return __func__;
}

const char *test_sum_dim1_keepdim() {
    tensor_t *t = tensor_alloc(2, (dim_sz_t[]){2, 3});
    float vals[] = {1,2,3,4,5,6};
    for (int i = 0; i < 6; i++) t->data[i] = vals[i];

    tensor_t *r = sum(t, 1, true);
    assert(r != NULL);
    assert(r->ndim == 2 && r->shape[0] == 2 && r->shape[1] == 1);
    assert(r->data[0] == 1+2+3);
    assert(r->data[1] == 4+5+6);

    tensor_free(t);
    tensor_free(r);

    return __func__;
}

const char *test_sum_negative_dim() {
    tensor_t *t = tensor_alloc(2, (dim_sz_t[]){2, 3});
    float vals[] = {1,2,3,4,5,6};
    for (int i = 0; i < 6; i++) t->data[i] = vals[i];

    tensor_t *r = sum(t, -1, false); // last dim
    assert(r != NULL);
    assert(r->ndim == 1 && r->shape[0] == 2);
    assert(r->data[0] == 1+2+3);
    assert(r->data[1] == 4+5+6);

    tensor_free(t);
    tensor_free(r);

    return __func__;
}

const char *test_sum_dim_out_of_range() {
    CHECK_ABORT({
        tensor_t *t = tensor_alloc(2, (dim_sz_t[]){2, 2});
        for (int i = 0; i < 4; i++) t->data[i] = i+1;
        sum(t, 2, false); // invalid dim, should hit assert and abort
    });

    return __func__;
}

const char *test_sum_dim4() {
    tensor_t *t = tensor_alloc(4, (dim_sz_t[]){2, 3, 2, 4});
    for (uint32_t i = 0; i < t->numel; i++) t->data[i] = i + 1;

    // comparing results with pytorch's
    {
        float expected[] = { 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72 };
        uint32_t numel = sizeof(expected) / sizeof(*expected);
        tensor_t *r = sum(t, 0, false);
        assert(r->numel == numel);
        for (uint32_t i = 0; i < numel; i++) assert(expected[i] == r->data[i]);
    }

    {
        float expected[] = { 27,  30,  33,  36,  39,  42,  45,  48,  99, 102, 105, 108, 111, 114, 117, 120 };
        uint32_t numel = sizeof(expected) / sizeof(*expected);
        tensor_t *r = sum(t, 1, false);
        assert(r->numel == numel);
        for (uint32_t i = 0; i < numel; i++) assert(expected[i] == r->data[i]);
    }

    {
        float expected[] = { 6,  8, 10, 12, 22, 24, 26, 28, 38, 40, 42, 44, 54, 56, 58, 60, 70, 72, 74, 76, 86, 88, 90, 92 };
        uint32_t numel = sizeof(expected) / sizeof(*expected);
        tensor_t *r = sum(t, 2, false);
        assert(r->numel == numel);
        for (uint32_t i = 0; i < numel; i++) assert(expected[i] == r->data[i]);
    }

    {
        float expected[] = { 10,  26,  42,  58,  74,  90, 106, 122, 138, 154, 170, 186 };
        uint32_t numel = sizeof(expected) / sizeof(*expected);
        tensor_t *r = sum(t, 3, false);
        assert(r->numel == numel);
        for (uint32_t i = 0; i < numel; i++) assert(expected[i] == r->data[i]);
    }

    return __func__;
}

const char *(*fnx[])(void) = {
    // test_transpose,
    // test_is_contiguous,
    test_reshape
    // test_sumall, test_sum_dim0, test_sum_dim1_keepdim, test_sum_negative_dim, test_sum_dim_out_of_range, test_sum_dim4,
};

int main(int argc, char **argv) {
    for (uint32_t i = 0; i < sizeof(fnx) / sizeof(*fnx); i++) printf("%s\n", fnx[i]());

    // tensor_t *t = tensor_alloc(3, (dim_sz_t[]){2, 3, 4});
    // for (uint32_t i = 0; i < t->numel; i++) t->data[i] = i + 1;

    // printf("t.shape = ");
    // tprint_shape(t->ndim, t->shape);
    // printf("\n");
    // printf("t.stride = ");
    // tprint_stride(t->ndim, t->stride);
    // printf("\n");
    // printf("t.data = \n");
    // tprint(t);

    // printf("\ntranspose(t)\n\n");
    // transpose(t, -1, -2);

    // printf("t.shape = ");
    // tprint_shape(t->ndim, t->shape);
    // printf("\n");
    // printf("t.stride = ");
    // tprint_stride(t->ndim, t->stride);
    // printf("\n");
    // printf("t.data = \n");
    // tprint(t);

    // printf("\n");
    // reshape(t, 6, (dim_sz_t[]){2, 1, 2, 2, 3, 1});
    // printf("t.shape = ");
    // tprint_shape(t->ndim, t->shape);
    // printf("\n");
    // printf("t.stride = ");
    // tprint_stride(t->ndim, t->stride);
    // printf("\n");
    // printf("t.data = \n");
    // tprint(t);

    return 0;
}