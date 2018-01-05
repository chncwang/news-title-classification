#include "n3ldg_cuda.h"
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include "cuPrintf.cuh"
#include "cuPrintf.cu"
#include "memory_pool.h"
#include "profiler.h"
#include "cnmem.h"

namespace n3ldg_cuda {

using std::cout;
using std::endl;

#define cuda_sqrt(x) sqrtf(x)
#define cuda_pow(x, y) powf(x, y)
#define cuda_tanh(x) tanhf(x)

#define KERNEL_LOG

#ifdef KERNEL_LOG
#define  KernelPrintLine(format, ...)\
{\
    cuPrintf("block:x=%d,y=%d thread:x=%d,y=%d "#format"\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,\
            __VA_ARGS__);\
}
#else
#define KernelPrintLine(format, ...)
#endif

constexpr int THREAD_COUNT_PER_BLOCK = 1024;
constexpr int BLOCK_COUNT = 56;

void CallCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
        abort();
    }
}

void CallCnmem(cnmemStatus_t status) {
    assert(status == CNMEM_STATUS_SUCCESS);
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
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(dtype*)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(dtype*),
                cudaMemcpyHostToDevice));
    this->len = len;
}

NumberPointerArray::~NumberPointerArray() {
    assert(value != NULL);
    CallCuda(MemoryPool::Ins().Free(value));
}

void NumberArray::init(dtype *host_arr, int len) {
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(dtype)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(dtype),
                cudaMemcpyHostToDevice));
    this->len = len;
}

NumberArray::~NumberArray() {
    assert(value != NULL);
    CallCuda(MemoryPool::Ins().Free(value));
}

void IntPointerArray::init(int **host_arr, int len) {
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(int*)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(int*),
                cudaMemcpyHostToDevice));
    this->len = len;
}

IntPointerArray::~IntPointerArray() {
    assert(value != NULL);
    CallCuda(MemoryPool::Ins().Free(value));
}

void IntArray::init(int *host_arr, int len) {
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(int)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(int),
                cudaMemcpyHostToDevice));
    this->len = len;
}

IntArray::~IntArray() {
    assert(value != NULL);
    CallCuda(MemoryPool::Ins().Free(value));
}

void Tensor1D::init(int dim) {
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, dim * sizeof(dtype)));
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
    CallCuda(MemoryPool::Ins().Free(value));
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
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, row * col * sizeof(dtype)));
    v = new dtype[row * col];
    this->row = row;
    this->col = col;
    this->size = row * col;
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
    CallCuda(MemoryPool::Ins().Free(value));
    delete [] v;
}

void Tensor2D::copyFromHostToDevice() {
    CallCuda(cudaMemcpy(value, v, size * sizeof(dtype), cudaMemcpyHostToDevice));
}

void Tensor2D::copyFromDeviceToHost() {
    CallCuda(cudaMemcpy(v, value, size * sizeof(dtype), cudaMemcpyDeviceToHost));
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
    //cudaSetDevice(1);

    cnmemDevice_t device;
    device.size = 10000000000;
    device.device = 1;
    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);

    CallCuda(cudaPrintfInit());
}

void EndCuda() {
    cudaPrintfEnd();
    Profiler::Ins().Print();
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
    int step = blockDim.x * gridDim.x;
    for (int i = index; i < len * count; i += step) {
        int count_i = i % count;
        int len_i = i / count;
        dtype result = cuda_tanh(src[i]);
        dest[count_i][len_i] = result;
        dest2[i] = result;
    }
}

void Tanh(const dtype *src, const std::vector<dtype*>& dest, dtype *dest2, int len) {
    int count = dest.size();
    NumberPointerArray dest_arr = ToNumberPointerArray(dest);
    int block_count = std::min((len * count - 1 + THREAD_COUNT_PER_BLOCK) /
        THREAD_COUNT_PER_BLOCK, BLOCK_COUNT);
    Tanh<<<block_count, THREAD_COUNT_PER_BLOCK>>>(src, dest_arr.value, dest2, count, len);
}

__global__ void KernelCopyForUniNodeForward(const dtype** xs, const dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int x_total_len = count * x_len;
    for (int i = index; i < x_total_len; i += step) {
        int len_i = i / count;
        int count_i = i % count;
        xs_dest[i] = xs[count_i][len_i];
    }
    for (int i = index; i < x_total_len + count * b_len; i += step) {
        int b_i = i - x_total_len;
        int len_i = b_i / count;
        b_dest[b_i] = b[len_i];
    }
}

void CopyForUniNodeForward(const std::vector<dtype*> &xs, const dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len) {
    int len = x_len + b_len;
    int block_count = std::min((count * len - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK, 56);
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
    CallCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, count, row, col, &alpha, x, count, W, col, &beta,
            y,
            count));
#else
    CallCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, count, row, col, &alpha, x, count, W, col, &beta,
            y,
            count));
#endif
}

__global__ void KernelVerify(dtype *host, dtype *device, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        dtype loss = host[index] - device[index];
        if (loss > 0.01 || loss < -0.01) {
            KernelPrintLine("KernelVerify: host:%f device:%f loss:%f",
                    host[index],
                    device[index],
                    loss);
        }
    }
}

void Verify(dtype *host, dtype *device, int len) {
    NumberArray arr;
    arr.init(host, len);
    int block_count = (len + THREAD_COUNT_PER_BLOCK - 1) /
        THREAD_COUNT_PER_BLOCK;
    KernelVerify<<<block_count, THREAD_COUNT_PER_BLOCK>>>(arr.value, device,
            len);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
}

cudaError_t MemoryPool::Malloc(void **p, int size) {
    CallCnmem(cnmemMalloc(p, size, NULL));
    return cudaSuccess;

    //return cudaMalloc(p, size);

//    bool found = false;
//    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
//        if (size == it->size) {
//            busy_blocks_.push_back(*it);
//            *p = it->p;
//            free_blocks_.erase(it);
//            found = true;
//            break;
//        }
//    }
//
//    cudaError_t status = cudaSuccess;
//    if (!found) {
//        status = cudaMalloc(p, size);
//        assert(status == cudaSuccess);
//        MemoryBlock block(*p, size);
//        busy_blocks_.push_back(block);
//    }
//
//    return status;
}

void MemoryPool::FreePool() {
    if (!busy_blocks_.empty()) {
        std::cout << "warning: busy_blocks_ not empty size:" << busy_blocks_.size() <<std::endl;
        for (MemoryBlock &b : busy_blocks_) {
            CallCuda(cudaFree(b.p));
        }
    }

    for (MemoryBlock &b : free_blocks_) {
        CallCuda(cudaFree(b.p));
    }
}

cudaError_t MemoryPool::Free(void *p) {
    CallCnmem(cnmemFree(p, NULL));

//    return cudaFree(p);

//    for (auto it = busy_blocks_.begin(); it != busy_blocks_.end(); ++it) {
//        if (p == it->p) {
//            free_blocks_.push_back(*it);
//            busy_blocks_.erase(it);
//            break;
//        }
//    }

    return cudaSuccess;
}

void Profiler::EndCudaEvent() {
    cudaDeviceSynchronize();
    EndEvent();
}

__global__ void KernelLtyForUniBackward(const dtype **ly, const dtype *ty,
        const dtype *y,
        dtype *lty,
        int count,
        int dim) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int step = blockDim.x * gridDim.x;
    int len = count * dim;
    for (int i = index; i < len; i += step) {
        int count_i = i % count;
        int dim_i = i / count;
        dtype tyi = ty[i];
        lty[i] = ly[count_i][dim_i] * (1 - tyi * tyi);
    }
}

void LtyForUniBackward(const std::vector<dtype*> &ly, const dtype *ty,
        const dtype *y,
        dtype *lty,
        int count,
        int dim) {
    int block_count = std::min(BLOCK_COUNT,
            (count * dim + THREAD_COUNT_PER_BLOCK - 1) /
            THREAD_COUNT_PER_BLOCK);
    NumberPointerArray ly_arr = ToNumberPointerArray(ly);
    KernelLtyForUniBackward<<<block_count, THREAD_COUNT_PER_BLOCK>>>(ly_arr,
            ty, y, lty, count, dim);
}

}
