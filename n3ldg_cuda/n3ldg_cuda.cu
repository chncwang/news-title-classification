#include "n3ldg_cuda.h"
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>

namespace n3ldg_cuda {

using std::cout;
using std::endl;

#define cuda_sqrt(x) sqrtf(x)
#define cuda_pow(x, y) powf(x, y)
#define cuda_tanh(x) tanh(x)

constexpr int THREAD_COUNT_PER_BLOCK = 1024;

void CallCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
        abort();
    }
}

void CallCublas(cublasStatus_t status) {
    assert(status == CUBLAS_STATUS_SUCCESS);
}

cublasHandle_t& GetCublasHandle() {
    static cublasHandle_t handle;
    static bool init;
    if (!init) {
        init = true;
        CallCublas(cublasCreate(&handle));
    }
    return handle;
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
#if N3LDG_DEBUG
   // CallCuda(cudaHostAlloc((void**)&value, dim * sizeof(dtype),
   //             cudaHostAllocDefault));
    CallCuda(cudaMalloc((void**)&value, dim * sizeof(dtype)));
#else
    CallCuda(cudaMalloc((void**)&value, dim * sizeof(dtype)));
#endif
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

void Tensor1D::copyFromHostToDevice() {
    assert(v != NULL);
    assert(value != NULL);
    CallCuda(cudaMemcpy(value, v, dim * sizeof(dtype), cudaMemcpyHostToDevice));
}

void Tensor1D::copyFromDeviceToHost() {
    CallCuda(cudaMemcpy(v, value, dim * sizeof(dtype), cudaMemcpyDeviceToHost));
}

void Tensor2D::init(int row, int col) {
#if N3LDG_DEBUG
   // CallCuda(cudaHostAlloc((void**)&value, row * col * sizeof(dtype),
   //             cudaHostAllocDefault));
    CallCuda(cudaMalloc((void**)&value, row * col * sizeof(dtype)));
#else
    CallCuda(cudaMalloc((void**)&value, row * col * sizeof(dtype)));
#endif
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

void Tensor2D::copyFromHostToDevice() {
    CallCuda(cudaMemcpy(value, v, size() * sizeof(dtype), cudaMemcpyHostToDevice));
}

void Tensor2D::copyFromDeviceToHost() {
    CallCuda(cudaMemcpy(v, value, size() * sizeof(dtype), cudaMemcpyDeviceToHost));
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

__global__ void KernelRescaleGrads(dtype **grads, int *lens,
        dtype max_scale) {
    __shared__ volatile dtype sum_temp[THREAD_COUNT_PER_BLOCK];
    __shared__ volatile bool is_last_block_in_x;

    if (threadIdx.x == 0) {
        is_last_block_in_x = false;
    }

    dtype *grad = grads[blockIdx.y];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) {
        block_nums_in_x[blockIdx.y] = 0;
        if (blockIdx.y == 0) {
            block_num_in_y = 0;
        }
    }
    int len = lens[blockIdx.y];
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
}

void RescaleGrads(std::vector<dtype *> &grads, const std::vector<int> &lens,
        dtype max_scale) {
    assert(grads.size() == lens.size());
    assert(grads.size() <= 100);
    int max_len = *std::max_element(lens.begin(), lens.end());
    std::cout << "max_len:" << max_len << std::endl;
    assert(max_len < THREAD_COUNT_PER_BLOCK * THREAD_COUNT_PER_BLOCK);

    int block_count = (max_len - 1 + THREAD_COUNT_PER_BLOCK) /
        THREAD_COUNT_PER_BLOCK;
    dim3 block_dim(block_count, grads.size(), 1);

    NumberPointerArray grads_arr = ToNumberPointerArray(grads);
    IntArray len_arr = ToIntArray(lens);

    KernelRescaleGrads<<<block_dim, THREAD_COUNT_PER_BLOCK>>>(grads_arr.value,
            len_arr.value, max_scale);
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

__global__ void PrintPointers(void **p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%p\n", p[i]);
    }
}

__global__ void PrintNums(dtype* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%f,", p[i]);
    }
    printf("\n");
}


void InitCuda() {
    cudaSetDevice(1);
}

__global__ void KernelCopyFromOneVectorToMultiVectors(const dtype *src,
        dtype *dest, int count, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len * count) {
        int count_i = index / len;
        int len_i = index % len;
        dest[count_i * len + len_i] = src[len_i];
    }
}

void CopyFromOneVectorToMultiVectors(const dtype *src, dtype *dest, int count, int len) {
    KernelCopyFromOneVectorToMultiVectors<<<
        (len * count - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK, THREAD_COUNT_PER_BLOCK>>>(
                src, dest, count, len);
}

__global__ void Tanh(const dtype *src, dtype**dest, dtype* dest2, int count, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len * count) {
        int count_i = index / len;
        int len_i = index % len;
        dtype result = cuda_tanh(src[index]);
        dest[count_i][len_i] = result;
        dest2[index] = result;
    }
}

void Tanh(const dtype *src, const std::vector<dtype*>& dest, dtype *dest2, int len) {
    int count = dest.size();
    NumberPointerArray dest_arr = ToNumberPointerArray(dest);
    Tanh<<<(len * count - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK,
    THREAD_COUNT_PER_BLOCK>>>(src, dest_arr.value, dest2, count, len);
}

__global__ void KernelCopyForUniNodeForward(const dtype** xs, const dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int x_total_len = count * x_len;
    if (index < x_total_len) {
        int count_i = index / x_len;
        int len_i = index % x_len;
        xs_dest[index] = xs[count_i][len_i];
    } else if (index < x_total_len + count * b_len) {
        int b_index = index - x_total_len;
        int len_i = b_index % b_len;
        b_dest[b_index] = b[len_i];
    }
}

void CopyForUniNodeForward(const std::vector<dtype*> &xs, const dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len) {
    int len = x_len + b_len;
    int block_count = (count * len - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK;
    NumberPointerArray xs_arr = ToNumberPointerArray(xs);
    KernelCopyForUniNodeForward<<<block_count, THREAD_COUNT_PER_BLOCK>>>((const dtype**)xs_arr.value,
            (const dtype*)b, xs_dest,
            b_dest,
            count,
            x_len,
            b_len);
}

void MatrixMultiplyMatrix(dtype *W, dtype *x, dtype *y, int row, int col, int count, bool useb) {
    cublasHandle_t &handle = GetCublasHandle();
    float alpha = 1;
    float beta = useb? 1 : 0;
#if USE_FLOAT
    CallCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row, count, col, &alpha, W, row, x, col, &beta,
            y,
            row));
#else
    CallCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row, count, col, &alpha, W, row, x, col, &beta,
            y,
            row));
#endif
}

}
