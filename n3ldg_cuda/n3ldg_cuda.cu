#include "n3ldg_cuda.h"
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace n3ldg_cuda {

using std::cout;
using std::endl;

#define cuda_sqrt(x) sqrtf(x)
#define cuda_pow(x, y) powf(x, y)

constexpr int THREAD_COUNT_PER_BLOCK = 1024;

void CallCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
        abort();
    }
}

NumberPointerArray ToNumberPointerArray(const std::vector<dtype*> &vec) {
    NumberPointerArray device_arr;
    device_arr.init(const_cast<dtype**>(vec.data()), vec.size());
    return device_arr;
}

IntPointerArray ToIntPointerArray(const std::vector<int*> &vec) {
    IntPointerArray device_arr;
    device_arr.init(const_cast<int**>(vec.data()), vec.size());
    return device_arr;
}

IntArray ToIntArray(const std::vector<int> vec) {
    IntArray device_arr;
    device_arr.init(const_cast<int*>(vec.data()), vec.size());
    return device_arr;
}

void NumberPointerArray::init(dtype **host_arr, int len) {
    CallCuda(cudaMalloc(&value, len * sizeof(dtype*)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(dtype*),
                cudaMemcpyHostToDevice));
    this->len = len;
}

NumberPointerArray::~NumberPointerArray() {
    assert(value != NULL);
    CallCuda(cudaFree(value));
}

void IntPointerArray::init(int **host_arr, int len) {
    CallCuda(cudaMalloc(&value, len * sizeof(int*)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(int*),
                cudaMemcpyHostToDevice));
    this->len = len;
}

IntPointerArray::~IntPointerArray() {
    assert(value != NULL);
    CallCuda(cudaFree(value));
}

void IntArray::init(int *host_arr, int len) {
    CallCuda(cudaMalloc(&value, len * sizeof(int)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(int),
                cudaMemcpyHostToDevice));
    this->len = len;
}

IntArray::~IntArray() {
    assert(value != NULL);
    CallCuda(cudaFree(value));
}

void Tensor1D::init(int dim) {
    CallCuda(cudaMalloc((void**)&value, dim * sizeof(dtype)));
    this->dim = dim;
    v = new dtype[dim];
    zero();
}

Tensor1D::Tensor1D(const Tensor1D &t) {
    dim = t.dim;
    memcpy(v, t.v, dim *sizeof(dtype));
    CallCuda(cudaMemcpy(value, t.value, dim * sizeof(dtype),
                cudaMemcpyDeviceToDevice));
}

Tensor1D::~Tensor1D() {
    assert(value != NULL && v != NULL);
    CallCuda(cudaFree(value));
    delete []v;
}

void Tensor2D::init(int row, int col) {
    CallCuda(cudaMalloc((void**)&value, row * col * sizeof(dtype)));
    v = new dtype[row * col];
    this->row = row;
    this->col = col;
    zero();
}

Tensor2D::Tensor2D(const Tensor2D &t) {
    row = t.row;
    col = t.col;
    memcpy(v, t.v, sizeof(dtype) * row * col);
    CallCuda(cudaMemcpy(value, t.value, sizeof(dtype) * row * col,
                cudaMemcpyDeviceToDevice));
}

Tensor2D::~Tensor2D() {
    assert(value != NULL && v != NULL);
    CallCuda(cudaFree(value));
    delete [] v;
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
    aux_square[index] = belta2 * aux_square[index] + (1 - belta2) *
        grad[index] * grad[index];

    dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2, iter + 1)) /
        (1 - cuda_pow(belta1, iter + 1));
    val[index] = val[index] - aux_mean[index] * lr_t /
        cuda_sqrt(aux_square[index] + eps);
}

void UpdateAdam(Tensor2D &val, Tensor2D &grad, Tensor2D &aux_mean,
        Tensor2D &aux_square, int &iter, dtype belta1, dtype belta2,
        dtype alpha, dtype reg, dtype eps) {
    int block_count = (val.row * val.col - 1 + THREAD_COUNT_PER_BLOCK) /
        THREAD_COUNT_PER_BLOCK;
    KernelUpdateAdam<<<block_count, THREAD_COUNT_PER_BLOCK>>>(val.value,
            grad.value, aux_mean.value, aux_square.value, val.row, val.col,
            iter, belta1, belta2, alpha, reg, eps);
    ++iter;
}

__device__ volatile dtype global_sum_temp[100][THREAD_COUNT_PER_BLOCK];
__device__ int block_nums_in_x[100];
__device__ int block_num_in_y;
__device__ volatile dtype global_sum;

__global__ void KernelUpdateAdamBatch(dtype **vals, dtype **grads,
        dtype ** aux_means,
        dtype **aux_squares,
        int *rows,
        int *cols,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        dtype max_scale) {
    __shared__ volatile dtype sum_temp[THREAD_COUNT_PER_BLOCK];
    __shared__ volatile bool is_last_block_in_x;

    if (threadIdx.x == 0) {
        is_last_block_in_x = false;
    }

    dtype *grad = grads[blockIdx.y];
    dtype row = rows[blockIdx.y];
    dtype col = cols[blockIdx.y];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) {
        block_nums_in_x[blockIdx.y] = 0;
        if (blockIdx.y == 0) {
            block_num_in_y = 0;
        }
    }
    int len = row * col;
    if (index < len) {
        dtype grad_val = grad[index];
        sum_temp[threadIdx.x] = grad_val * grad_val;
    } else {
        sum_temp[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int i = (THREAD_COUNT_PER_BLOCK >> 1); i > 0; i >>=1) {
        if (threadIdx.x < i) {
            sum_temp[threadIdx.x] += sum_temp[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_sum_temp[blockIdx.y][blockIdx.x] = sum_temp[0];
        if (atomicAdd(block_nums_in_x + blockIdx.y, 1) == gridDim.x - 1) {
            is_last_block_in_x = true;
        }
    }
    __syncthreads();

    if (is_last_block_in_x) {
        if (threadIdx.x < gridDim.x) {
            sum_temp[threadIdx.x] = global_sum_temp[blockIdx.y][threadIdx.x];
        } else {
            sum_temp[threadIdx.x] = 0;
        }
        __syncthreads();

        for (int i = (THREAD_COUNT_PER_BLOCK >> 1); i > 0; i >>=1) {
            if (threadIdx.x < i) {
                sum_temp[threadIdx.x] += sum_temp[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            global_sum_temp[0][blockIdx.y] = sum_temp[0];
            if (atomicAdd(&block_num_in_y, 1) == gridDim.y - 1) {
                dtype sum = 0;
                for (int i = 0; i<gridDim.y; ++i) {
                    sum += global_sum_temp[0][i];
                }
                global_sum = sum;
            }
        }
    }
    __syncthreads();
    int local_global_sum = global_sum;

    assert(local_global_sum < 1e20);
    dtype norm = cuda_sqrt(local_global_sum);
    if (max_scale > 0 && norm > max_scale) {
        dtype scale = max_scale / norm;
        grad[index] *= scale;
    }

    dtype* val = vals[blockIdx.y];
    if (index < len) {
        if (col > 1 && row > 1) {
            grad[index] = grad[index] + reg * val[index];
        }
        __syncthreads();

        dtype *aux_mean = aux_means[blockIdx.y];
        dtype *aux_square = aux_squares[blockIdx.y];
        aux_mean[index] = belta1 * aux_mean[index] + (1-belta1) * grad[index];
        aux_square[index] = belta2 * aux_square[index] + (1 - belta2) *
            grad[index] * grad[index];

        dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2, iters[blockIdx.y] + 1)) /
            (1 - cuda_pow(belta1, iters[blockIdx.y] + 1));
        val[index] = val[index] - aux_mean[index] * lr_t /
            cuda_sqrt(aux_square[index] + eps);
    }
}

void UpdateAdamBatch(std::vector<dtype *> &vals, std::vector<dtype *> &grads,
        std::vector<dtype *> &aux_mean,
        std::vector<dtype *> &aux_square,
        const std::vector<int> &rows,
        const std::vector<int> &cols,
        const std::vector<int> &iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        dtype max_scale) {
    assert(grads.size() == vals.size() && vals.size() == aux_mean.size()
            && vals.size() == aux_square.size() &&
            vals.size() == iters.size() && vals.size() == rows.size()
            && vals.size() == cols.size());
    assert(vals.size() <= 100);
    int max_col = *std::max_element(cols.begin(), cols.end());
    int max_row = *std::max_element(rows.begin(), rows.end());
    assert(max_col * max_row <
            THREAD_COUNT_PER_BLOCK * THREAD_COUNT_PER_BLOCK);

    int block_count = (max_col *max_row - 1 + THREAD_COUNT_PER_BLOCK) /
        THREAD_COUNT_PER_BLOCK;
    dim3 block_dim(block_count, vals.size(), 1);

    NumberPointerArray vals_arr = ToNumberPointerArray(vals);
    NumberPointerArray grads_arr = ToNumberPointerArray(grads);
    NumberPointerArray aux_mean_arr = ToNumberPointerArray(aux_mean);
    NumberPointerArray aux_square_arr = ToNumberPointerArray(aux_square);
    IntArray iters_arr = ToIntArray(iters);
    IntArray row_arr = ToIntArray(rows);
    IntArray col_arr = ToIntArray(cols);

    KernelUpdateAdamBatch<<<block_dim, THREAD_COUNT_PER_BLOCK>>>(
            vals_arr.value,
            grads_arr.value,
            aux_mean_arr.value,
            aux_square_arr.value,
            row_arr.value,
            col_arr.value,
            iters_arr.value,
            belta1,
            belta2,
            alpha,
            reg,
            eps,
            max_scale);
}

void Random(dtype *v, int len, dtype bound) {
    dtype *mem = (dtype*)malloc(len * sizeof(dtype));
    assert(mem != NULL);
    dtype min = -bound, max = bound;
    for (int i = 0; i < len; i++) {
        mem[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
    }

    CallCuda(cudaMemcpy(v, mem, len * sizeof(dtype), cudaMemcpyHostToDevice));

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
    int block_count = (len - 1 + THREAD_COUNT_PER_BLOCK) /
        THREAD_COUNT_PER_BLOCK;
    KernelZero<<<block_count, THREAD_COUNT_PER_BLOCK>>>(v, len);
}

}
