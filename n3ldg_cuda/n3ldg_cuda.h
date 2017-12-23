#ifndef N3LDG_CUDA_N3LDG_CUDA_H
#define N3LDG_CUDA_N3LDG_CUDA_H

#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace n3ldg_cuda {

struct Tensor1D {
    dtype *value = NULL;
    int len = 0;

    void init(int len);
    ~Tensor1D();

    void random(dtype bound);
};

struct Tensor2D {
    dtype *value = NULL;
    int row = 0;
    int col = 0;

    void init(int row, int col);
    ~Tensor2D();

    void random(dtype bound);
    void zero();
};

void UpdateAdam(Tensor2D &val, Tensor2D &grad, Tensor2D &aux_mean,
        Tensor2D &aux_square, int &iter, dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps);

#endif
