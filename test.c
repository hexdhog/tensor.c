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

const char *test_sumall() {
    uint16_t shape[] = {2, 3};
    tensor_t *t = tensor_alloc(2, shape);
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
    uint16_t shape[] = {2, 3};
    tensor_t *t = tensor_alloc(2, shape);
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
    uint16_t shape[] = {2, 3};
    tensor_t *t = tensor_alloc(2, shape);
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
    uint16_t shape[] = {2, 3};
    tensor_t *t = tensor_alloc(2, shape);
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
        tensor_t *t = tensor_alloc(2, (uint16_t[]){2, 2});
        for (int i = 0; i < 4; i++) t->data[i] = i+1;
        sum(t, 2, false); // invalid dim, should hit assert and abort
    });

    return __func__;
}

const char *(*fnx[])(void) = {
    test_sumall, test_sum_dim0, test_sum_dim1_keepdim, test_sum_negative_dim, test_sum_dim_out_of_range
};

int main(int argc, char **argv) {
    for (uint32_t i = 0; i < sizeof(fnx) / sizeof(*fnx); i++) {
        printf("%s\n", fnx[i]());
    }

    return 0;
}
