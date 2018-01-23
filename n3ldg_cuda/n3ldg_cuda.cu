#include "n3ldg_cuda.h"
#include <array>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include "cuPrintf.cuh"
#include "cuPrintf.cu"
#include "memory_pool.h"
#include <curand.h>
#include <curand_kernel.h>
#include "profiler.h"
#include "cnmem.h"
#include <string>
#include <cstdint>

namespace n3ldg_cuda {

using std::cout;
using std::endl;

#if USE_FLOAT
#define cuda_sqrt(x) sqrtf(x)
#define cuda_pow(x, y) powf(x, y)
#define cuda_tanh(x) tanhf(x)
#define cuda_exp(x) __expf(x)
#else
#define cuda_sqrt(x) sqrt(x)
#define cuda_pow(x, y) pow(x, y)
#define cuda_tanh(x) tanh(x)
#define cuda_exp(x) exp(x)
#endif

#define KERNEL_LOG

#ifdef KERNEL_LOG
#define  KernelPrintLine(format, ...)\
{\
    cuPrintf("block:x=%d,y=%d thread:x=%d,y=%d "#format"\n", blockIdx.x,\
            blockIdx.y, threadIdx.x, threadIdx.y,__VA_ARGS__);\
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

void CallCurand(curandStatus status) {
    assert(status == CURAND_STATUS_SUCCESS);
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

NumberPointerPointerArray ToNumberPointerPointerArray(
        const std::vector<dtype**> &vec) {
    NumberPointerPointerArray device_arr;
    device_arr.init(const_cast<dtype***>(vec.data()), vec.size());
    return device_arr;
}

IntPointerArray ToIntPointerArray(const std::vector<int*> &vec) {
    IntPointerArray device_arr;
    device_arr.init(const_cast<int**>(vec.data()), vec.size());
    return device_arr;
}

IntArray ToIntArray(const std::vector<int> &vec) {
    IntArray device_arr;
    device_arr.init(const_cast<int*>(vec.data()), vec.size());
    return device_arr;
}

BoolArray ToBoolArray(const bool *vec, int len) {
    BoolArray device_arr;
    device_arr.init(const_cast<bool*>(vec), len);
    return device_arr;
}

void NumberPointerArray::init(dtype **host_arr, int len) {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(dtype*)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(dtype*),
                cudaMemcpyHostToDevice));
    this->len = len;
}

void Memcpy(dtype *dest, dtype*src, int size, cudaMemcpyKind kind) {
    CallCuda(cudaMemcpy(dest, src, size, kind));
}

NumberPointerArray::~NumberPointerArray() {
    CallCuda(MemoryPool::Ins().Free(value));
}

void NumberPointerPointerArray::init(dtype ***host_arr, int len) {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(dtype**)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(dtype*),
                cudaMemcpyHostToDevice));
    this->len = len;
}

NumberPointerPointerArray::~NumberPointerPointerArray() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void NumberArray::init(dtype *host_arr, int len) {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(dtype)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(dtype),
                cudaMemcpyHostToDevice));
    this->len = len;
}

NumberArray::~NumberArray() {
    CallCuda(MemoryPool::Ins().Free(value));
}

void DeviceInt::init() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, sizeof(int)));
}

void DeviceInt::copyFromDeviceToHost() {
    CallCuda(cudaMemcpy(&v, value, sizeof(int), cudaMemcpyDeviceToHost));
}

DeviceInt::~DeviceInt() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void IntPointerArray::init(int **host_arr, int len) {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(int*)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(int*),
                cudaMemcpyHostToDevice));
    this->len = len;
}

IntPointerArray::~IntPointerArray() {
    CallCuda(MemoryPool::Ins().Free(value));
}

void IntArray::init(int *host_arr, int len) {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(int)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(int),
                cudaMemcpyHostToDevice));
    this->len = len;
}

void IntArray::init(int len) {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(int)));
    this->len = len;
}

IntArray::~IntArray() {
    CallCuda(MemoryPool::Ins().Free(value));
}

void BoolArray::init(bool *host_arr, int len) {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(bool)));
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(bool),
                cudaMemcpyHostToDevice));
    this->len = len;
}

void BoolArray::copyFromHost(bool *host_arr) {
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(bool),
                cudaMemcpyHostToDevice));
}

BoolArray::~BoolArray() {
    CallCuda(MemoryPool::Ins().Free(value));
}

void Tensor1D::init(int dim) {
    initOnDevice(dim);
    v = new dtype[dim];
    zero();
}

void Tensor1D::initOnDevice(int dim) {
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, dim * sizeof(dtype)));
    this->dim = dim;
}

Tensor1D::Tensor1D(const Tensor1D &t) {
    dim = t.dim;
    memcpy(v, t.v, dim *sizeof(dtype));
    CallCuda(cudaMemcpy(value, t.value, dim * sizeof(dtype),
                cudaMemcpyDeviceToDevice));
}

Tensor1D::~Tensor1D() {
    CallCuda(MemoryPool::Ins().Free(value));
    if (v != NULL) {
        delete []v;
    }
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
    initOnDevice(row, col);
    v = new dtype[row * col];
    zero();
}

void Tensor2D::initOnDevice(int row, int col) {
    CallCuda(MemoryPool::Ins().Malloc((void**)&value,
                row * col * sizeof(dtype)));
    this->row = row;
    this->col = col;
    this->size = row * col;
}

Tensor2D::Tensor2D(const Tensor2D &t) {
    row = t.row;
    col = t.col;
    memcpy(v, t.v, sizeof(dtype) * row * col);
    CallCuda(cudaMemcpy(value, t.value, sizeof(dtype) * row * col,
                cudaMemcpyDeviceToDevice));
}

Tensor2D::~Tensor2D() {
    CallCuda(MemoryPool::Ins().Free(value));
    if (v != NULL) {
        delete [] v;
    }
}

void Tensor2D::copyFromHostToDevice() {
    CallCuda(cudaMemcpy(value, v, size * sizeof(dtype), cudaMemcpyHostToDevice));
}

void Tensor2D::copyFromDeviceToHost() {
    CallCuda(cudaMemcpy(v, value, size * sizeof(dtype), cudaMemcpyDeviceToHost));
}

void Assert(bool v) {
    if (!v) exit(1);
}

void *Malloc(int size) {
    void *p;
    CallCuda(cudaMalloc(&p, size));
    return p;
}

__device__ void DeviceAtomicAdd(float* address, float value) {
    float old = value;  
    float new_old;
    do {
        new_old = atomicExch(address, 0.0f);
        new_old += old;
    } while ((old = atomicExch(address, new_old))!=0.0f);
};

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

__device__ int DeviceDefaultIndex() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int DeviceDefaultStep() {
    return gridDim.x * blockDim.x;
}

__device__ dtype DeviceAbs(dtype d) {
    return d > 0 ? d : -d;
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

__global__ void PrintNums(const dtype* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%f\n", p[i]);
    }
}


void InitCuda() {
#if DEVICE_MEMORY == 0
    cnmemDevice_t device;
    device.size = 2000000000;
    device.device = 1;
    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
#else
    CallCuda(cudaSetDevice(1));
#endif
    CallCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    CallCuda(cudaPrintfInit());
}

void EndCuda() {
    cudaPrintfEnd();
    Profiler::Ins().Print();
}

__global__ void KernelCopyFromOneVectorToMultiVectors(const dtype *src,
        dtype *dest, int count, int len) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len * count; i += step) {
        int count_i = i / len;
        int len_i = i % len;
        dest[count_i * len + len_i] = src[len_i];
    }
}

void CopyFromOneVectorToMultiVectors(const dtype *src, dtype *dest, int count,
        int len) {
    int block_count = (len * count - 1 + THREAD_COUNT_PER_BLOCK) /
        THREAD_COUNT_PER_BLOCK;
    block_count = std::min(block_count, BLOCK_COUNT);
    KernelCopyFromOneVectorToMultiVectors<<<
        block_count, THREAD_COUNT_PER_BLOCK>>>(src, dest, count, len);
}

__global__ void KernelCopyFromOneVectorToMultiVectors(const dtype *src,
        dtype **dest, int count, int len) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * len; i += step) {
        int count_i = i % count;
        int len_i = i / count;
        dest[count_i][len_i] = src[i];
    }
}

void CopyFromOneVectorToMultiVectors(const dtype *src,
        const std::vector<dtype*> &dest, int count, int len) {
    NumberPointerArray dest_arr = ToNumberPointerArray(dest);
    int block_count = (len * count - 1 + THREAD_COUNT_PER_BLOCK) /
        THREAD_COUNT_PER_BLOCK;
    block_count = std::min(block_count, BLOCK_COUNT);
    KernelCopyFromOneVectorToMultiVectors<<<block_count,
    THREAD_COUNT_PER_BLOCK>>>(src, dest_arr.value, count, len);
}

__global__ void KernelTanh(const dtype *src, dtype**dest, dtype* dest2,
        int count, int len, bool is_being_trained, dtype drop_factor,
        const dtype *drop_mask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    __syncthreads();

    for (int i = index; i < len * count; i += step) {
        int count_i = i % count;
        int len_i = i / count;
        dtype result = cuda_tanh(src[i]);
        if (is_being_trained) {
            if (drop_mask[i] <= drop_factor) {
                dest[count_i][len_i] = 0.0f;
                dest2[i] = result;
            } else {
                dest[count_i][len_i] = result;
                dest2[i] = result;
            }
        } else {
            dest[count_i][len_i] = result * (1 - drop_factor);
            dest2[i] = result;
        }
    }
}

__global__ void KernelCountDrop(dtype *y, int dim) {
    int count = 0;
    for (int i = 0; i < dim; ++i) {
        if (y[i] > -0.0001 && y[i] < 0.0001) {
            ++count;
        }
    }
    KernelPrintLine("drop count:%d", count);
}

void Tanh(const dtype *src, const std::vector<dtype*>& dest, dtype *dest2,
        int len, bool is_being_trained, dtype drop_factor,
        const dtype *drop_mask) {
    if (drop_factor < 0) {
        drop_factor = 0;
    }
    int count = dest.size();
    NumberPointerArray dest_arr = ToNumberPointerArray(dest);
    int block_count = std::min((len * count - 1 + THREAD_COUNT_PER_BLOCK) /
        THREAD_COUNT_PER_BLOCK, BLOCK_COUNT);
    KernelTanh<<<block_count, THREAD_COUNT_PER_BLOCK>>>(src, dest_arr.value,
            dest2, count, len, is_being_trained, drop_factor, drop_mask);
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
    int b_total_len = count * b_len;
    for (int i = index; i < x_total_len + b_total_len; i += step) {
        if (i < x_total_len) {
            int len_i = i / count;
            int count_i = i % count;
            xs_dest[i] = xs[count_i][len_i];
        } else {
            int b_i = i - x_total_len;
            int len_i = b_i / count;
            b_dest[b_i] = b[len_i];
        }
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

void MatrixMultiplyMatrix(dtype *W, dtype *x, dtype *y, int row, int col,
        int count, bool useb, bool should_x_transpose,
        bool should_W_transpose) {
    cublasHandle_t &handle = GetCublasHandle();
    float alpha = 1;
    float beta = useb? 1 : 0;
    cublasOperation_t x_op = should_x_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldx = should_x_transpose ? col : count;
    cublasOperation_t W_op = should_W_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldw = should_W_transpose ? row : col;
#if USE_FLOAT
    CallCublas(cublasSgemm(handle, x_op, W_op, count, row, col,
                &alpha, x, ldx, W, ldw, &beta, y, count));
#else
    CallCublas(cublasDgemm(handle, x_op, W_op, count, row, col,
                &alpha, x, ldx, W, ldw, &beta, y, count));
#endif
}

__global__ void KernelVerify(dtype *host, dtype *device, int len,
        const char *message, bool *success) {
    int index = DeviceDefaultIndex();
    if (index < len) {
        dtype loss = host[index] - device[index];
        if (DeviceAbs(loss) > 0.001) {
            *success = false;
            printf("KernelVerify %s: host:%f device:%f loss:%f\n",
                    message,
                    host[index],
                    device[index],
                    loss);
            KernelPrintLine("KernelVerify: host:%f device:%f loss:%f",
                    host[index],
                    device[index],
                    loss);
        }
    }
}

bool Verify(dtype *host, dtype *device, int len, const char* message) {
    NumberArray arr;
    arr.init(host, len);
    int block_count = (len + THREAD_COUNT_PER_BLOCK - 1) /
        THREAD_COUNT_PER_BLOCK;
    char *m = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&m,
                (strlen(message) + 1) * sizeof(char)));
    CallCuda(cudaMemcpy(m, message,
                (strlen(message) + 1) * sizeof(char), cudaMemcpyHostToDevice));
    bool success = true;
    bool *dev_success = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&dev_success, 8 * sizeof(bool)));
    CallCuda(cudaMemcpy(dev_success, &success, sizeof(bool),
                cudaMemcpyHostToDevice));
    KernelVerify<<<block_count, THREAD_COUNT_PER_BLOCK>>>(arr.value, device,
            len, m, dev_success);
    CallCuda(cudaMemcpy(&success, dev_success, sizeof(bool),
                cudaMemcpyDeviceToHost));
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    return success;
}

__global__ void KernelVerify(bool *host, bool *device, int len,
        const char *message, bool *success) {
    int index = DeviceDefaultIndex();
    if (index < len) {
        if (host[index] != device[index]) {
            *success = false;
            printf("KernelVerify %s: host:%d device:%d \n", message,
                    host[index],
                    device[index]);
            KernelPrintLine("KernelVerify: host:%d device:%d", host[index],
                    device[index]);
        }
    }
}

bool Verify(bool *host, bool *device, int len, const char* message) {
    BoolArray arr;
    arr.init(host, len);
    int block_count = (len + THREAD_COUNT_PER_BLOCK - 1) /
        THREAD_COUNT_PER_BLOCK;
    char *m = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&m,
                (strlen(message) + 1) * sizeof(char)));
    CallCuda(cudaMemcpy(m, message,
                (strlen(message) + 1) * sizeof(char), cudaMemcpyHostToDevice));
    bool success = true;
    bool *dev_success = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&dev_success, 8 * sizeof(bool)));
    CallCuda(cudaMemcpy(dev_success, &success, sizeof(bool),
                cudaMemcpyHostToDevice));
    KernelVerify<<<block_count, THREAD_COUNT_PER_BLOCK>>>(arr.value, device,
            len, m, dev_success);
    CallCuda(cudaMemcpy(&success, dev_success, sizeof(bool),
                cudaMemcpyDeviceToHost));
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    return success;
}

cudaError_t MemoryPool::Malloc(void **p, int size) {
    assert(*p == NULL);
#if DEVICE_MEMORY == 0
    CallCnmem(cnmemMalloc(p, size, NULL));
    return cudaSuccess;
#elif DEVICE_MEMORY == 1
    return cudaMalloc(p, size);
#else
    //std::cout << "free size:" << free_blocks_.size() << " busy size:" <<
    //    busy_blocks_.size() << std::endl;
    Profiler &profiler = Profiler::Ins();
    //profiler.BeginEvent("malloc");
    int min_size = 1000000000;
    std::list<MemoryBlock>::iterator min_it = free_blocks_.end();
    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
        if (size <= it->size && min_size > it->size) {
            min_size = it->size;
            min_it = it;
        }
    }

    cudaError_t status = cudaSuccess;
    if (min_it != free_blocks_.end()) {
        //std::cout << "cache hit" << std::endl;
        busy_blocks_.push_back(*min_it);
        *p = min_it->p;
        free_blocks_.erase(min_it);
    } else {
        //std::cout << "no block, malloc" << std::endl;
        status = cudaMalloc(p, size);
        CallCuda(status);
        MemoryBlock block(*p, size);
        busy_blocks_.push_back(block);
    }

    //profiler.EndEvent();
    return status;
#endif
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
#if DEVICE_MEMORY == 0
    CallCnmem(cnmemFree(p, NULL));
#elif DEVICE_MEMORY == 1
    return cudaFree(p);
#else
    for (auto it = busy_blocks_.end() - 1; it != busy_blocks_.begin() - 1;
            --it) {
        if (p == it->p) {
            free_blocks_.push_back(*it);
            busy_blocks_.erase(it);
            break;
        }
    }

    return cudaSuccess;
#endif
}

void Profiler::EndCudaEvent() {
    cudaDeviceSynchronize();
    EndEvent();
}

__global__ void KernelCalculateLtyForUniBackward(const dtype *const*ly,
        const dtype *ty,
        const dtype *y,
        const dtype *drop_mask,
        dtype drop_factor,
        dtype *lty,
        int count,
        int dim) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = count * dim;
    for (int i = index; i < len; i += step) {
        int count_i = i % count;
        int dim_i = i / count;
        dtype yi = y[i];
        if (drop_mask[i] <= drop_factor) {
            lty[i] = 0.0f;
        } else {
            lty[i] = ly[count_i][dim_i] * (1 - yi * yi);
        }
    }
}

void CalculateLtyForUniBackward(const std::vector<dtype*> &ly, const dtype *ty,
        const dtype *y,
        const dtype *drop_mask,
        dtype drop_factor,
        dtype *lty,
        int count,
        int dim) {
    if (drop_factor < 0) {
        drop_factor = 0;
    }
    NumberPointerArray ly_arr = ToNumberPointerArray(ly);
    int block_count = std::min(BLOCK_COUNT, (count * dim +
                THREAD_COUNT_PER_BLOCK - 1) / THREAD_COUNT_PER_BLOCK);
    KernelCalculateLtyForUniBackward<<<block_count,
        THREAD_COUNT_PER_BLOCK>>>(ly_arr.value, ty, y, drop_mask, drop_factor,
                lty, count, dim);
}

__global__ void KernelCalculateLyForLinearBackward(const dtype *const*ly_vec,
        dtype *ly, int count, int dim) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = count * dim;
    for (int i = index; i < len; i += step) {
        int count_i = i % count;
        int dim_i = i / count;
        ly[i] = ly_vec[count_i][dim_i];
    }
}

void CalculateLyForLinearBackward(const std::vector<dtype*> &ly_vec, dtype *ly,
        int count, int dim) {
    NumberPointerArray ly_arr = ToNumberPointerArray(ly_vec);
    int block_count = std::min(BLOCK_COUNT, (count * dim +
                THREAD_COUNT_PER_BLOCK - 1) / THREAD_COUNT_PER_BLOCK);
    KernelCalculateLyForLinearBackward<<<block_count,
        THREAD_COUNT_PER_BLOCK>>>(ly_arr.value, ly, count, dim);
}

__device__ int global_block_count[1000000];
__global__ void KernelAddLtyToParamBiasAndAddLxToInputLossesForUniBackward(
        const dtype *lty,
        const dtype *lx,
        dtype *b,
        dtype **losses,
        int count,
        int out_dim,
        int in_dim,
        dtype *block_sums) {
    __shared__ volatile dtype shared_arr[THREAD_COUNT_PER_BLOCK];

    int count_i = blockIdx.y * blockDim.x + threadIdx.x;
    int dim_i = blockIdx.x;
    if (dim_i < out_dim) {
        if (threadIdx.x == 0 && blockIdx.y == 0) {
            global_block_count[dim_i] = 0;
        }
        int lty_index = dim_i * count + count_i;
        shared_arr[threadIdx.x] = count_i < count ? lty[lty_index] : 0.0f;
        __syncthreads();

        for (int i = (THREAD_COUNT_PER_BLOCK >> 1); i > 0; i>>=1) {
            if (threadIdx.x < i) {
                shared_arr[threadIdx.x] += shared_arr[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            block_sums[gridDim.y * blockIdx.x + blockIdx.y] = shared_arr[0];
            if (atomicAdd(global_block_count + dim_i, 1) == gridDim.y - 1) {
                dtype sum = 0.0;
                for (int i = 0; i < gridDim.y; ++i) {
                    sum += block_sums[gridDim.y * blockIdx.x + i];
                }
                DeviceAtomicAdd(b + dim_i, sum);
            }
        }
    } else {
        if (count_i < count) {
            dim_i -= out_dim;
            int lx_index = dim_i * count + count_i;
            DeviceAtomicAdd(losses[count_i] + dim_i, lx[lx_index]);
        }
    }
}

void AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(const dtype *lty,
        const dtype *lx, dtype *b, std::vector<dtype*> &losses, int count,
        int out_dim, int in_dim) {
    int block_y = (count - 1 + THREAD_COUNT_PER_BLOCK) /
        THREAD_COUNT_PER_BLOCK;
    dim3 block_dim(out_dim + in_dim, block_y, 1);
    NumberPointerArray loss_arr;
    loss_arr.init(losses.data(), count);
    Tensor1D block_sums;
    block_sums.init(block_y * out_dim);
    KernelAddLtyToParamBiasAndAddLxToInputLossesForUniBackward<<<block_dim,
        THREAD_COUNT_PER_BLOCK>>>(lty, lx, b, loss_arr.value, count, out_dim,
                in_dim, block_sums.value);
    //cudaPrintfDisplay(stdout, true);
}

constexpr int MAX_BATCH_COUNT = 1000000;

__global__ void KernelInitCurandStates(curandState_t *states) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (int i = index; i < MAX_BATCH_COUNT; i += step) {
        curand_init(0, i, 0, &states[i]);
    }
}

curandState_t *GetCurandStates() {
    static curandState_t *states;
    if (states == NULL) {
        MemoryPool &pool = MemoryPool::Ins();
        CallCuda(pool.Malloc((void**)&states, sizeof(curandState_t) *
                    MAX_BATCH_COUNT));
        KernelInitCurandStates<<<BLOCK_COUNT, THREAD_COUNT_PER_BLOCK>>>(
                states);
    }
    return states;
}

curandGenerator_t &GetGenerator() {
    static curandGenerator_t gen;
    static bool init;
    if (!init) {
        CallCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CallCurand(curandSetPseudoRandomGeneratorSeed(gen, 0));
        init = true;
    }
    return gen;
}

void CalculateDropoutMask(dtype drop_factor, int count, int dim, dtype* mask) {
    curandGenerator_t &gen = GetGenerator();
    CallCurand(curandGenerateUniform(gen, mask, count * dim));
}

__global__ void KernelConcatForward(dtype ***ins, const int *in_offsets,
        const dtype *drop_mask,
        dtype drop_factor,
        dtype** outs,
        int count,
        int in_count,
        int out_dim) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < out_dim * count; i += step) {
        int out_dim_i = i % out_dim;
        int count_i = i / out_dim;
        dtype dropout = drop_factor > 0 ?
            drop_mask[out_dim_i * count + count_i] : 1;
        if (dropout <= drop_factor) {
            outs[count_i][out_dim_i] = 0.0f;
        } else {
            int offset_j = 0;
            for (int j = 0; j < in_count; ++j) {
                if (out_dim_i < in_offsets[j]) {
                    break;
                }
                offset_j = j;
        }
            int in_dim_i = out_dim_i - in_offsets[offset_j];
            outs[count_i][out_dim_i] = ins[count_i][offset_j][in_dim_i];
        }
    }
}

void ConcatForward(const std::vector<dtype**> &ins, const int *in_offsets,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &outs,
        int count,
        int in_count,
        int out_dim) {
    assert(drop_factor < 1);
    if (drop_factor < 0) {
        drop_factor = 0;
    }
    NumberPointerPointerArray ins_arr = ToNumberPointerPointerArray(ins);
    NumberPointerArray out_arr = ToNumberPointerArray(outs);
    int len = count * out_dim;
    int block_count = std::min(BLOCK_COUNT,
            (len - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK);
    KernelConcatForward<<<block_count, THREAD_COUNT_PER_BLOCK>>>(ins_arr.value,
            in_offsets, drop_mask, drop_factor, out_arr.value, count, in_count,
            out_dim);
}

__global__ void KernelConcatBackward(const dtype** out_losses,
        const int *in_offsets,
        const dtype *drop_mask,
        dtype drop_factor,
        dtype*** in_losses,
        int count,
        int in_count,
        int out_dim) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < out_dim * count; i += step) {
        int out_dim_i = i % out_dim;
        int count_i = i / out_dim;
        dtype dropout = drop_factor > 0 ?
            drop_mask[out_dim_i * count + count_i] : 1;
        if (dropout > drop_factor) {
            int offset_j = 0;
            for (int j = 0; j < in_count; ++j) {
                if (out_dim_i < in_offsets[j]) {
                    break;
                }
                offset_j = j;
            }
            //KernelPrintLine("offset_j:%d", offset_j);
            int in_dim_i = out_dim_i - in_offsets[offset_j];
            //KernelPrintLine("in_dim_i:%d out_dim_i:%d", in_dim_i, out_dim_i);
//            in_losses[count_i][offset_j][in_dim_i] +=
//                out_losses[count_i][out_dim_i];
            DeviceAtomicAdd(in_losses[count_i][offset_j] + in_dim_i,
                    out_losses[count_i][out_dim_i]);
        }
    }
}

void ConcatBackward(const std::vector<dtype*> &out_losses,
        const int *in_offsets,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype**> in_losses,
        int count,
        int in_count,
        int out_dim) {
    assert(drop_factor < 1);
    if (drop_factor < 0) {
        drop_factor = 0;
    }
    NumberPointerArray out_loss_arr = ToNumberPointerArray(out_losses);
    NumberPointerPointerArray in_loss_arr =
        ToNumberPointerPointerArray(in_losses);
    int len = count * out_dim;
    int block_count = std::min(BLOCK_COUNT,
            (len - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK);
    //std::cout << "len:" << len << " block_count:" << block_count << std::endl;
    KernelConcatBackward<<<block_count, THREAD_COUNT_PER_BLOCK>>>(
            const_cast<const dtype**>(out_loss_arr.value), in_offsets,
            drop_mask, drop_factor, in_loss_arr.value, count, in_count,
            out_dim);
}

__global__ void KernelMemset(dtype *p, int len, dtype value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i+= step) {
        p[i] = value;
    }
}

void Memset(dtype *p, int len, dtype value) {
    int block_count = std::min(BLOCK_COUNT,
            (len - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK);
    KernelMemset<<<block_count, THREAD_COUNT_PER_BLOCK>>>(p, len, value);
}

__global__ void KernelMemset(bool *p, int len, bool value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i+= step) {
        p[i] = value;
    }
}

void Memset(bool *p, int len, bool value) {
    int block_count = std::min(BLOCK_COUNT,
            (len - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK);
    KernelMemset<<<block_count, THREAD_COUNT_PER_BLOCK>>>(p, len, value);
}

__global__ void KernelLookupForward(const int *xids, const dtype *vocabulary,
        const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int dim,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype dropout = drop_factor > 0 ?
            drop_mask[dim_i * count + count_i] : 1;
        if (drop_factor < dropout) {
            int xid = xids[count_i];
            if (xid >= 0) {
                int voc_i = xid * dim + dim_i;
                vals[count_i][dim_i] = vocabulary[voc_i];
            } else {
                vals[count_i][dim_i] = 0.0f;
            }
        } else {
            vals[count_i][dim_i] = 0.0f;
        }
    }
}

__global__ void Print2DNums(dtype ** nums, int count, int dim) {
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < dim; ++j) {
            printf("%f,", nums[i][j]);
        }
        printf("\n");
    }
}

void LookupForward(const std::vector<int> &xids, const dtype *vocabulary,
        const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int dim,
        std::vector<dtype*> &vals) {
    if (drop_factor < 0) {
        drop_factor = 0;
    }
    int block_count = std::min(BLOCK_COUNT, (count * dim - 1 +
                THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK);
    Profiler &profiler = Profiler::Ins();
    IntArray xid_arr = ToIntArray(xids);
    NumberPointerArray val_arr = ToNumberPointerArray(vals);
    KernelLookupForward<<<block_count, THREAD_COUNT_PER_BLOCK>>>(xid_arr.value,
            vocabulary, drop_mask, drop_factor,  count, dim,
            const_cast<dtype**>(val_arr.value));
}

__global__ void KernelLookupBackward(const int *xids, int unknown_id,
        bool fine_tune,
        const dtype** losses,
        const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int dim,
        dtype *grad,
        bool *indexers) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int xid = xids[count_i];
        if (xid == unknown_id || fine_tune) {
            assert(xid >= 0);
            if (dim_i == 0) {
                indexers[xid] = true;
            }
            dtype dropout = drop_factor > 0 ?
                drop_mask[dim_i * count + count_i] : 1;
            if (drop_factor < dropout) {
                DeviceAtomicAdd(grad + xid * dim + dim_i,
                        losses[count_i][dim_i]);
            }
        }
    }
}

void LookupBackward(const std::vector<int> &xids, int unknown_id,
        bool fine_tune,
        const std::vector<dtype*> &losses,
        const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int dim,
        dtype *grad,
        bool *indexers) {
    int block_count = std::min((count * dim - 1 + THREAD_COUNT_PER_BLOCK) /
            THREAD_COUNT_PER_BLOCK, BLOCK_COUNT);
    IntArray xid_arr = ToIntArray(xids);
    NumberPointerArray loss_arr = ToNumberPointerArray(losses);
    KernelLookupBackward<<<block_count, THREAD_COUNT_PER_BLOCK>>>(
            const_cast<const int *>(xid_arr.value),
            unknown_id,
            fine_tune,
            const_cast<const dtype**>(loss_arr.value),
            drop_mask,
            drop_factor,
            count,
            dim,
            grad,
            indexers);
}

__global__ void KernelMaxPoolForward(const dtype ***ins, int count,
        int *in_counts,
        int dim,
        int* hit_inputs,
        dtype** outs) {
    __shared__ volatile dtype shared_arr[THREAD_COUNT_PER_BLOCK];
    __shared__ volatile dtype shared_indexers[THREAD_COUNT_PER_BLOCK];
    int batch_i = blockIdx.y;
    int in_count = in_counts[batch_i];
    int in_count_i = threadIdx.x;
    int dim_i = blockIdx.x;
    shared_arr[threadIdx.x] = in_count_i < in_count ?
        ins[batch_i][in_count_i][dim_i] : -INFINITY;
    shared_indexers[threadIdx.x] = threadIdx.x;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0;i >>=1) {
        if (threadIdx.x < i) {
            int plus_i = threadIdx.x + i;
            if (shared_arr[threadIdx.x] < shared_arr[plus_i]) {
                shared_arr[threadIdx.x] = shared_arr[plus_i];
                shared_indexers[threadIdx.x] = plus_i;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        hit_inputs[batch_i * dim + dim_i] = shared_indexers[0];
        outs[batch_i][dim_i] = shared_arr[0];
    }
}

void MaxPoolForward(const std::vector<dtype**> &ins, int count,
        const std::vector<int> &in_counts,
        int dim,
        int *hit_inputs,
        std::vector<dtype*> &outs) {
    IntArray in_count_arr = ToIntArray(in_counts);
    NumberPointerPointerArray in_arr = ToNumberPointerPointerArray(ins);
    NumberPointerArray out_arr = ToNumberPointerArray(outs);

    int max_in_count = *std::max_element(in_counts.begin(), in_counts.end());
    int thread_count = 8;
    while (max_in_count > thread_count) {
        thread_count <<= 1;
    }

    dim3 block_dim(dim, count, 1);
    KernelMaxPoolForward<<<block_dim, thread_count>>>(
            const_cast<const dtype***>(in_arr.value),
            count,
            in_count_arr.value,
            dim,
            hit_inputs,
            out_arr.value);
}

__global__ void KernelMaxPoolBackward(const dtype ** losses,
        const int *hit_inputs, int count, int dim, dtype ***in_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int input_i = hit_inputs[i];
        DeviceAtomicAdd(in_losses[count_i][input_i] + dim_i,
                losses[count_i][dim_i]);
    }
}

void MaxPoolBackward(const std::vector<dtype*> &losses, const int *hit_inputs,
        int count, 
        int dim,
        std::vector<dtype**> &in_losses) {
    NumberPointerArray loss_arr = ToNumberPointerArray(losses);
    NumberPointerPointerArray in_loss_arr = ToNumberPointerPointerArray(
            in_losses);
    int block_count = (count * dim - 1 + THREAD_COUNT_PER_BLOCK) /
        THREAD_COUNT_PER_BLOCK;
    block_count = std::min(block_count, BLOCK_COUNT);
    KernelMaxPoolBackward<<<block_count, THREAD_COUNT_PER_BLOCK>>>(
            const_cast<const dtype**>(loss_arr.value),
            hit_inputs,
            count,
            dim,
            in_loss_arr.value);
}

__global__ void KernelSoftMaxLoss(const dtype **vals, dtype **losses,
        int *correct_count, int *answers, int batchsize, int count, int dim) {
    volatile __shared__ int opt_label;
    volatile extern __shared__ char shared_arr[];
    volatile dtype * shared_val = (dtype*)(shared_arr);
    volatile int32_t *max_indexes =
        (int32_t*)(shared_arr + sizeof(dtype) * blockDim.x);
    volatile dtype * scores = (dtype*)(shared_arr + (sizeof(dtype) +
                sizeof(int32_t)) * blockDim.x);
    volatile dtype * scores_sum = (dtype*)(shared_arr + (2 * sizeof(dtype) +
                sizeof(int32_t)) * blockDim.x);
    int dim_i = threadIdx.x;
    int count_i = blockIdx.x;
    //printf("count_i:%d\n", count_i);
    if (count_i == 0 && dim_i == 0) {
        *correct_count = 0;
    }
    shared_val[dim_i] = dim_i < dim ? vals[count_i][dim_i] : -INFINITY;
    //printf("shared_val:%f dim_i:%d\n", shared_val[dim_i], dim_i);
    max_indexes[dim_i] = dim_i;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (shared_val[threadIdx.x + i] > shared_val[threadIdx.x]) {
            shared_val[threadIdx.x] = shared_val[threadIdx.x + i];
            max_indexes[threadIdx.x] = max_indexes[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        opt_label = max_indexes[0];
        //printf("opt_label:%d\n", opt_label);
        if (answers[count_i] == opt_label) {
            atomicAdd(correct_count, 1);
        }
    }
    __syncthreads();

    //printf("opt_label:%d\n", opt_label);
    dtype max_score = vals[count_i][opt_label];
    //printf("max_score:%f\n", max_score);
    dtype score = dim_i < dim ? cuda_exp(vals[count_i][dim_i] - max_score) :
        0.0f;
    //printf("score:%f\n", score);
    scores[dim_i] = score;
    scores_sum[dim_i] = score;

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        scores_sum[threadIdx.x] = scores_sum[threadIdx.x] +
            scores_sum[threadIdx.x + i];
        __syncthreads();
    }

    if (dim_i < dim) {
        //printf("count_i:%d dim_i:%d scores_sum[0]:%f answer:%d batchsize:%d\n", count_i, dim_i, scores_sum[0], answers[count_i], batchsize);
        losses[count_i][dim_i] = (scores[dim_i] / scores_sum[0] -
                (dim_i == answers[count_i] ? 1 : 0)) / batchsize;
    }
}

void SoftMaxLoss(const std::vector<dtype*> &vals, std::vector<dtype*> &losses,
        int *correct_count,
        const std::vector<int> &answers,
        int batchsize,
        int count,
        int dim) {
    int thread_count = 1;
    while (dim > thread_count) {
        thread_count <<= 1;
    }
    NumberPointerArray val_arr = ToNumberPointerArray(vals);
    NumberPointerArray loss_arr = ToNumberPointerArray(losses);
    IntArray answer_arr = ToIntArray(answers);
    KernelSoftMaxLoss<<<count, thread_count,
        (sizeof(int32_t) + 3 * sizeof(dtype)) * thread_count>>>(
            const_cast<const dtype **>(val_arr.value),
            const_cast<dtype **>(loss_arr.value),
            correct_count,
            answer_arr.value,
            batchsize,
            count,
            dim);
}

}
