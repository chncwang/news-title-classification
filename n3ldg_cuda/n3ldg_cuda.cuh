#ifndef N3LDG_CUDA_N3LDG_CUDA_H
#define N3LDG_CUDA_N3LDG_CUDA_H

#include <iostream>
#include <cassert>

namespace n3ldg_cuda {

typedef float dtype;

void CallCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
        abort();
    }
}

struct Tensor1D {
    dtype *value_ = NULL;
    int len_ = 0;

    void init(int len) {
        CallCuda(cudaMalloc((void**)&value_, len * sizeof(dtype)));
        len_ = len;
    }

    ~Tensor1D() {
        assert(value_ != NULL);
        CallCuda(cudaFree(value_));
    }
};

struct Tensor2D {
    dtype *value_ = NULL;
    int row_ = 0;
    int col_ = 0;

    void init(int row, int col) {
        CallCuda(cudaMalloc((void**)&value_, row * col * sizeof(dtype)));
        row_ = row;
        col_ = col;
    }
};

#endif
