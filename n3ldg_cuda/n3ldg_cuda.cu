#include "n3ldg_cuda.cuh"
#include <crand>

namespace n3ldg_cuda {

typedef float dtype;

#define cuda_sqrt(x) sqrtf(x)
#define cuda_pow(x, y) powf(x, y)

constexpr int THREAD_COUNT_PER_BLOCK = 1024;
constexpr int MAX_BLOCK_COUNT = 56;

void CallCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
        abort();
    }
}

void Tensor1D::init(int len) {
    CallCuda(cudaMalloc((void**)&value, len * sizeof(dtype)));
    this->len = len;
}

Tensor1D::~Tensor1D() {
    assert(value != NULL);
    CallCuda(cudaFree(value));
}

void Tensor1D::random(dtype bound) {
    Random(value, len, bound);
}

void Tensor2D::init(int row, int col) {
    CallCuda(cudaMalloc((void**)&value, row * col * sizeof(dtype)));
    this->row = row;
    this->col = col;
}

Tensor2D::~Tensor2D() {
    assert(value != NULL);
    CallCuda(cudaFree(value));
}

void Tensor2D::random(dtype bound) {
    Random(value, row * col, bound);
}

__global__ void KernelUpdateAdam(dtype *val,  dtype *grad,
        dtype *aux_mean, dtype *aux_square, int row, int col, int iter, dtype belta1,
        dtype belta2, dtype alpha, dtype reg, dtype eps) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int len = row * col;
    if (index >= len) {
        return;
    }
    if (col > 1 && row > 1) {
        grad[index] = grad[index] + reg * val[index];
    }
    __syncthreads();

    aux_mean[index] = belta1 * aux_mean[index] + (1-belta1) * grad[index];
    aux_square[index] = belta2 * aux_square[index] + (1 - belta2) * grad[index] * grad[index];

    dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2, iter + 1)) / (1 - cuda_pow(belta1, iter + 1));
    val[index] = val[index] - aux_mean[index] * lr_t / cuda_sqrt(aux_square[index] + eps);
}

void UpdateAdam(Tensor2D &val, Tensor2D &grad, Tensor2D &aux_mean,
        Tensor2D &aux_square, int &iter, dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
    int block_count = (val.row * val.col - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK;
    KernelUpdateAdam<<<block_count, THREAD_COUNT_PER_BLOCK>>>(val.value, grad.value, aux_mean.value,
            aux_square.value, val.row, val.col, iter, belta1, belta2, alpha, reg, eps);
    ++iter;
}

void Random(dtype *v, int len, dtype bound) {
    float *mem = malloc(len * sizeof(dtype));
    assert(mem != NULL);
    dtype min = -bound, max = bound;
    for (int i = 0; i < len; i++) {
        mem[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
    }

    CallCuda(cudaMemcpy(v, mem, cudaMemcpyHostToDevice));

    free(mem);
}

__global__ void KernelZero(dtype *v, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) {
        return;
    }
    v[index] = 0;
}

void Zero(dtype *v, int len) {
    int block_count = (len - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK;
    KernelZero<<<block_count, THREAD_COUNT_PER_BLOCK>>>(v, len);
}

}
