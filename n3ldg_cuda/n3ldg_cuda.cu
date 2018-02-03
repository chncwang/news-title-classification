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
#include "cnmem.h"
#include <string>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <thread>

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

constexpr int TPB = 1024;
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

NumberPointerArray::~NumberPointerArray() {
    CallCuda(MemoryPool::Ins().Free(value));
}

void PageLockedNumberPointerArray::init(dtype **host_arr, int len) {
    if (value != NULL) {
        CallCuda(PageLockedMemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(PageLockedMemoryPool::Ins().Malloc((void**)&value,
                len * sizeof(dtype*)));
    memcpy(value, host_arr, len * sizeof(dtype*));
    this->len = len;
}

PageLockedNumberPointerArray::~PageLockedNumberPointerArray() {
    if (value != NULL) {
        CallCuda(PageLockedMemoryPool::Ins().Free(value));
    }
}

dtype **PageLockedNumberPointerArray::GetDevicePointer() const {
    dtype **device_p;
    CallCuda(cudaHostGetDevicePointer((void**)&device_p, (void*)value, 0));
    return device_p;
}

void Memcpy(dtype *dest, dtype*src, int size, cudaMemcpyKind kind) {
    CallCuda(cudaMemcpy(dest, src, size, kind));
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

void NumberArray::init(int len) {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, len * sizeof(dtype)));
    this->len = len;
}

void NumberArray::init(dtype *host_arr, int len) {
    init(len);
    CallCuda(cudaMemcpy(value, host_arr, len * sizeof(dtype),
                cudaMemcpyHostToDevice));
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

void DeviceInt::copyFromHostToDevice() {
    CallCuda(cudaMemcpy(value, &v, sizeof(int), cudaMemcpyHostToDevice));
}

DeviceInt::~DeviceInt() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void DeviceNumber::init() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, sizeof(int)));
}

void DeviceNumber::copyFromDeviceToHost() {
    CallCuda(cudaMemcpy(&v, value, sizeof(dtype), cudaMemcpyDeviceToHost));
}

DeviceNumber::~DeviceNumber() {
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

void PageLockedIntArray::init(int *host_arr, int len) {
    if (value != NULL) {
        CallCuda(PageLockedMemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(PageLockedMemoryPool::Ins().Malloc((void**)&value,
                len * sizeof(int)));
    memcpy(value, host_arr, len * sizeof(int));
    this->len = len;
}

void PageLockedIntArray::init(int len) {
    if (value != NULL) {
        CallCuda(PageLockedMemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(PageLockedMemoryPool::Ins().Malloc((void**)&value,
                len * sizeof(int)));
    this->len = len;
}

PageLockedIntArray::~PageLockedIntArray() {
    CallCuda(PageLockedMemoryPool::Ins().Free(value));
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

void BoolArray::copyToHost(bool *host_arr) {
    CallCuda(cudaMemcpy(host_arr, value, len * sizeof(bool),
                cudaMemcpyDeviceToHost));
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
#if TEST_CUDA
    if (!v) {
        abort();
        exit(1);
    }
#endif
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

int DefaultBlockCount(int len) {
    int block_count = (len - 1 + TPB) /
        TPB;
    return std::min(block_count, BLOCK_COUNT);
}

int DefaultBlockCountWithoutLimit(int len) {
    return (len - 1 + TPB) / TPB;
}

__global__ void KernelZero(dtype *v, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) {
        return;
    }
    v[index] = 0;
}

void Zero(dtype *v, int len) {
    int block_count = (len - 1 + TPB) /
        TPB;
    KernelZero<<<block_count, TPB>>>(v, len);
}

__global__ void PrintPointers(void **p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%p\n", p[i]);
    }
}

__global__ void KernelPrintNums(const dtype* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%f\n", p[i]);
    }
}

void PrintNums(const dtype* p, int len) {
    KernelPrintNums<<<1, 1>>>(p, len);
    cudaDeviceSynchronize();
}

__global__ void KernelPrintInts(const int* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%d\n", p[i]);
    }
}

void PrintInts(const int* p, int len) {
    KernelPrintInts<<<1, 1>>>(p, len);
    cudaDeviceSynchronize();
}

void InitCuda() {
    CallCuda(cudaSetDeviceFlags(cudaDeviceMapHost));

#if DEVICE_MEMORY == 0
    cnmemDevice_t device;
    device.size = 2000000000;
    device.device = 0;
    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
#else
    CallCuda(cudaSetDevice(0));
#endif
    CallCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    CallCuda(cudaPrintfInit());
}

void EndCuda() {
    cudaPrintfEnd();
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
    int block_count = (len * count - 1 + TPB) / TPB;
    block_count = std::min(block_count, BLOCK_COUNT);
    KernelCopyFromOneVectorToMultiVectors<<<
        block_count, TPB>>>(src, dest, count, len);
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
    NumberPointerArray dest_arr;
    dest_arr.init((dtype**)dest.data(), dest.size());
    int block_count = (len * count - 1 + TPB) / TPB;
    block_count = std::min(block_count, BLOCK_COUNT);
    KernelCopyFromOneVectorToMultiVectors<<<block_count, TPB>>>(src,
            dest_arr.value, count, len);
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
    PageLockedNumberPointerArray dest_arr;
    dest_arr.init((dtype**)dest.data(), dest.size());
    int block_count = std::min((len * count - 1 + TPB) / TPB, BLOCK_COUNT);
    KernelTanh<<<block_count, TPB>>>(src, dest_arr.value,
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
    int block_count = std::min((count * len - 1 + TPB) / TPB, 56);
    PageLockedNumberPointerArray xs_arr;
    xs_arr.init((dtype**)xs.data(), xs.size());
    NumberPointerArray arr;
    arr.init((dtype**)xs_arr.value, xs.size());
    KernelCopyForUniNodeForward<<<block_count, TPB>>>(
            (const dtype**)arr.value,
            (const dtype*)b, xs_dest,
            b_dest,
            count,
            x_len,
            b_len);
    //std::cout << "hello world!" << std::endl;
    //std::this_thread::sleep_for(std::chrono::microseconds(1));
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
    int block_count = (len + TPB - 1) / TPB;
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
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
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
    int block_count = (len + TPB - 1) / TPB;
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
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
    CallCuda(cudaMemcpy(&success, dev_success, sizeof(bool),
                cudaMemcpyDeviceToHost));
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    return success;
}

__global__ void KernelVerify(int *host, int *device, int len,
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

bool Verify(int *host, int *device, int len, const char* message) {
    IntArray arr;
    arr.init(host, len);
    int block_count = (len + TPB - 1) / TPB;
    char *m = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&m,
                (strlen(message) + 1) * sizeof(char)));
    CallCuda(cudaMemcpy(m, message,
                (strlen(message) + 1) * sizeof(char), cudaMemcpyHostToDevice));
    bool success = true;
    bool *dev_success = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&dev_success, sizeof(bool)));
    CallCuda(cudaMemcpy(dev_success, &success, sizeof(bool),
                cudaMemcpyHostToDevice));
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
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

    return status;
#endif
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

cudaError_t PageLockedMemoryPool::Malloc(void **p, int size) {
    assert(*p == NULL);
    //std::cout << "free size:" << free_blocks_.size() << " busy size:" <<
    //    busy_blocks_.size() << std::endl;
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
        status = cudaHostAlloc(p, size, cudaHostAllocWriteCombined);
        CallCuda(status);
        MemoryBlock block(*p, size);
        busy_blocks_.push_back(block);
    }

    return status;
}

cudaError_t PageLockedMemoryPool::Free(void *p) {
//#if DEVICE_MEMORY == 1
//    return cudaFreeHost(p);
//#else
    for (auto it = busy_blocks_.end() - 1; it != busy_blocks_.begin() - 1;
            --it) {
        if (p == it->p) {
            free_blocks_.push_back(*it);
            busy_blocks_.erase(it);
            break;
        }
    }

    return cudaSuccess;
//#endif
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
    NumberPointerArray ly_arr;
    ly_arr.init((dtype**)ly.data(), ly.size());
    int block_count = std::min(BLOCK_COUNT, (count * dim + TPB - 1) / TPB);
    KernelCalculateLtyForUniBackward<<<block_count, TPB>>>(ly_arr.value, ty, y,
            drop_mask, drop_factor, lty, count, dim);
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
    NumberPointerArray ly_arr;
    ly_arr.init((dtype**)ly_vec.data(), ly_vec.size());
    int block_count = std::min(BLOCK_COUNT, (count * dim + TPB - 1) / TPB);
    KernelCalculateLyForLinearBackward<<<block_count,
        TPB>>>(ly_arr.value, ly, count, dim);
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
    __shared__ volatile dtype shared_arr[TPB];

    int count_i = blockIdx.y * blockDim.x + threadIdx.x;
    int dim_i = blockIdx.x;
    if (dim_i < out_dim) {
        if (threadIdx.x == 0 && blockIdx.y == 0) {
            global_block_count[dim_i] = 0;
        }
        int lty_index = dim_i * count + count_i;
        shared_arr[threadIdx.x] = count_i < count ? lty[lty_index] : 0.0f;
        __syncthreads();

        for (int i = (TPB >> 1); i > 0; i>>=1) {
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
    int block_y = (count - 1 + TPB) / TPB;
    dim3 block_dim(out_dim + in_dim, block_y, 1);
    NumberPointerArray loss_arr;
    loss_arr.init(losses.data(), count);
    Tensor1D block_sums;
    block_sums.init(block_y * out_dim);
    KernelAddLtyToParamBiasAndAddLxToInputLossesForUniBackward<<<block_dim,
        TPB>>>(lty, lx, b, loss_arr.value, count, out_dim,
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
        KernelInitCurandStates<<<BLOCK_COUNT, TPB>>>( states);
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

__global__ void KernelConcatForward(const void *graph, const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int in_count,
        int out_dim) {
    dtype **outs = (dtype**)graph;
    int offset = 2 * count * sizeof(dtype*);
    dtype **ins = (dtype**)((char*)graph + offset);
    offset += 2 * count * in_count * sizeof(dtype*);
    int32_t *in_dims = (int32_t*)((char*)graph + offset);

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
            int in_dim_sum = 0;
            int last_in_dim_sum;
            int offset_j = 0;
            for (int j = 0; j < in_count; ++j) {
                last_in_dim_sum = in_dim_sum;
                in_dim_sum += in_dims[j];
                offset_j = j;
                if (out_dim_i < in_dim_sum) {
                    break;
                }
            }
            int in_dim_i = out_dim_i - last_in_dim_sum;
            dtype v = ins[count_i * in_count + offset_j][in_dim_i];
            outs[count_i][out_dim_i] = v;
        }
    }
}

void ConcatForward(const void *graph, const dtype *drop_mask,
        dtype drop_factor, int count, int in_count, int out_dim) {
    assert(drop_factor < 1);
    if (drop_factor < 0) {
        drop_factor = 0;
    }
    int len = count * out_dim;
    int block_count = std::min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    KernelConcatForward<<<block_count, TPB>>>(graph, drop_mask, drop_factor,
            count, in_count, out_dim);
}

__global__ void KernelConcatBackward(const void* graph, const dtype *drop_mask,
        dtype drop_factor, int count, int in_count, int out_dim) {
    graph = (char*)graph + count * sizeof(dtype*);
    dtype **out_losses = (dtype**)graph;
    graph = (char*)graph + count * (1 + in_count) * sizeof(dtype*);
    dtype ** in_losses = (dtype**)graph;
    graph = (char*)graph + count * in_count * sizeof(dtype*);
    int32_t *in_dims = (int*)graph;

    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < out_dim * count; i += step) {
        int out_dim_i = i % out_dim;
        int count_i = i / out_dim;
        dtype dropout = drop_factor > 0 ?
            drop_mask[out_dim_i * count + count_i] : 1;
        if (dropout > drop_factor) {
            int in_dim_sum = 0;
            int last_in_dim_sum;
            int offset_j = 0;
            for (int j = 0; j < in_count; ++j) {
                last_in_dim_sum = in_dim_sum;
                in_dim_sum += in_dims[j];
                offset_j = j;
                if (out_dim_i < in_dim_sum) {
                    break;
                }
            }
            int in_dim_i = out_dim_i - last_in_dim_sum;
            DeviceAtomicAdd(in_losses[count_i * in_count + offset_j] +
                    in_dim_i, out_losses[count_i][out_dim_i]);
        }
    }
}

void ConcatBackward(const void *graph, const dtype *drop_mask,
        dtype drop_factor, int count, int in_count, int out_dim) {
    assert(drop_factor < 1);
    if (drop_factor < 0) {
        drop_factor = 0;
    }
    int len = count * out_dim;
    int block_count = std::min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    KernelConcatBackward<<<block_count, TPB>>>( graph, drop_mask, drop_factor,
            count, in_count, out_dim);
}

__global__ void KernelMemset(dtype *p, int len, dtype value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i+= step) {
        p[i] = value;
    }
}

void Memset(dtype *p, int len, dtype value) {
    int block_count = std::min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    KernelMemset<<<block_count, TPB>>>(p, len, value);
}

__global__ void KernelMemset(bool *p, int len, bool value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i+= step) {
        p[i] = value;
    }
}

void Memset(bool *p, int len, bool value) {
    int block_count = std::min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    KernelMemset<<<block_count, TPB>>>(p, len, value);
}

void *Malloc(int size) {
    void *p;
    CallCuda(cudaMalloc(&p, size));
    return p;
}

void Memcpy(void *dest, void *src, int size, cudaMemcpyKind kind) {
    CallCuda(cudaMemcpy(dest, src, size, kind));
}

__global__ void KernelBatchMemset(dtype **p, int count, int dim, dtype value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count ; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        p[count_i][dim_i] = value;
    }
}

void BatchMemset(const std::vector<dtype*> &vec, int count, int dim,
        dtype value) {
    int block_count = (count * dim -1 + TPB) / TPB;
    block_count = std::min(block_count, BLOCK_COUNT);
    NumberPointerArray vec_arr;
    vec_arr.init((dtype**)vec.data(), vec.size());
    KernelBatchMemset<<<block_count, TPB>>>(vec_arr.value, count, dim, value);
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
    int block_count = std::min(BLOCK_COUNT, (count * dim - 1 + TPB) / TPB);
    IntArray xid_arr;
    xid_arr.init((int*)xids.data(), xids.size());
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    KernelLookupForward<<<block_count, TPB>>>(xid_arr.value, vocabulary,
            drop_mask, drop_factor,  count, dim,
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
    int block_count = std::min((count * dim - 1 + TPB) / TPB, BLOCK_COUNT);
    PageLockedIntArray pl_arr;
    pl_arr.init((int*)xids.data(), xids.size());
    IntArray xid_arr;
    xid_arr.init((int*)pl_arr.value, xids.size());
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size());
    KernelLookupBackward<<<block_count, TPB>>>(
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
    __shared__ volatile extern dtype pool_shared_arr[];
    volatile dtype* shared_indexers = pool_shared_arr + blockDim.x;
    int batch_i = blockIdx.y;
    int in_count = in_counts[batch_i];
    int in_count_i = threadIdx.x;
    int dim_i = blockIdx.x;
    pool_shared_arr[threadIdx.x] = in_count_i < in_count ?
        ins[batch_i][in_count_i][dim_i] : -INFINITY;
    shared_indexers[threadIdx.x] = threadIdx.x;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0;i >>=1) {
        if (threadIdx.x < i) {
            int plus_i = threadIdx.x + i;
            if (pool_shared_arr[threadIdx.x] < pool_shared_arr[plus_i]) {
                pool_shared_arr[threadIdx.x] = pool_shared_arr[plus_i];
                shared_indexers[threadIdx.x] = shared_indexers[plus_i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        hit_inputs[batch_i * dim + dim_i] = shared_indexers[0];
        outs[batch_i][dim_i] = pool_shared_arr[0];
    }
}

void MaxPoolForward(const std::vector<dtype**> &ins, int count,
        const std::vector<int> &in_counts,
        int dim,
        int *hit_inputs,
        std::vector<dtype*> &outs) {
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());
    NumberPointerPointerArray in_arr;
    in_arr.init((dtype***)ins.data(), ins.size());
    NumberPointerArray out_arr;
    out_arr.init((dtype**)outs.data(), outs.size());

    int max_in_count = *std::max_element(in_counts.begin(), in_counts.end());
    int thread_count = 8;
    while (max_in_count > thread_count) {
        thread_count <<= 1;
    }

    dim3 block_dim(dim, count, 1);
    KernelMaxPoolForward<<<block_dim, thread_count, thread_count * 2 *
        sizeof(dtype)>>>(const_cast<const dtype***>(in_arr.value), count,
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
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size());
    NumberPointerPointerArray in_loss_arr;
    in_loss_arr.init((dtype***)in_losses.data(), in_losses.size());
    int block_count = (count * dim - 1 + TPB) / TPB;
    block_count = std::min(block_count, BLOCK_COUNT);
    KernelMaxPoolBackward<<<block_count, TPB>>>(
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
    if (count_i == 0 && dim_i == 0) {
        *correct_count = 0;
    }
    shared_val[dim_i] = dim_i < dim ? vals[count_i][dim_i] : -INFINITY;
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
        if (answers[count_i] == opt_label) {
            atomicAdd(correct_count, 1);
        }
    }
    __syncthreads();

    dtype max_score = vals[count_i][opt_label];
    dtype score = dim_i < dim ? cuda_exp(vals[count_i][dim_i] - max_score) :
        0.0f;
    scores[dim_i] = score;
    scores_sum[dim_i] = score;

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        scores_sum[threadIdx.x] = scores_sum[threadIdx.x] +
            scores_sum[threadIdx.x + i];
        __syncthreads();
    }

    if (dim_i < dim) {
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
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size());
    IntArray answer_arr;
    answer_arr.init((int*)answers.data(), answers.size());
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

__global__ void KernelSquareSum(const dtype *v, int len, dtype *global_sum,
        int *block_counter, dtype *result) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    int index = DeviceDefaultIndex();
    if (index == 0) {
        *block_counter = 0;
    }
    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = 0.0f;
        is_last_block = false;
    }
    if (index < len) {
        shared_sum[threadIdx.x] = v[index] * v[index];
    } else {
        shared_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = shared_sum[0];
        if (atomicAdd(block_counter, 1) == gridDim.x - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += global_sum[i];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *result = shared_sum[0];
        }
    }
}

dtype SquareSum(const dtype *v, int len) {
    int block_count = DefaultBlockCountWithoutLimit(len);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelSquareSum<<<block_count, TPB>>>(v, len,
            global_sum.value, block_counter.value, result.value);
    result.copyFromDeviceToHost();
    return result.v;
}

__global__ void KernelSquareSum(const dtype *v, const bool *indexers,
        int count,
        int dim,
        dtype *global_sum,
        int *block_counter,
        dtype *result) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    int index = DeviceDefaultIndex();
    if (index == 0) {
        *block_counter = 0;
    }
    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = 0.0f;
        is_last_block = false;
    }
    int count_i = index / dim;
    if (index < count * dim && indexers[count_i]) {
        shared_sum[threadIdx.x] = v[index] * v[index];
    } else {
        shared_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = shared_sum[0];
        if (atomicAdd(block_counter, 1) == gridDim.x - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += global_sum[i];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *result = shared_sum[0];
        }
    }
}

dtype SquareSum(const dtype *v, const bool *indexers, int count, int dim) {
    int block_count = DefaultBlockCountWithoutLimit(count * dim);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelSquareSum<<<block_count, TPB>>>(v, indexers,
            count, dim, global_sum.value, block_counter.value, result.value);
    result.copyFromDeviceToHost();
    return result.v;
}

__global__ void KernelRescale(dtype *v, int len, dtype scale) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i += step) {
        v[i] *= scale;
    }
}

void Rescale(dtype *v, int len, dtype scale) {
    int block_count = DefaultBlockCount(len);
    KernelRescale<<<block_count, TPB>>>(v, len, scale);
}

__global__ void KernelUpdateAdam(dtype *val, dtype *grad, int row, int col,
        dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        dtype x) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        if (row > 1 && col > 1) {
            grad[i] += val[i] * reg;
        }
        aux_mean[i] = belta1 * aux_mean[i] + (1 - belta1) * grad[i];
        aux_square[i] = belta2 * aux_square[i] + (1 - belta2) * grad[i] *
            grad[i];
        dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2, iter + 1)) * x;
        dtype square_plus_eps = aux_square[i] + eps;
        val[i] = val[i] - aux_mean[i] * lr_t / cuda_sqrt(square_plus_eps);
    }
}

void UpdateAdam(dtype *val, dtype *grad, int row, int col, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    dtype x = 1.0f / (1 - pow(belta1, iter + 1));
    KernelUpdateAdam<<<block_count, TPB>>>(val, grad, row, col, aux_mean,
            aux_square,
            iter,
            belta1,
            belta2,
            alpha,
            reg,
            eps,
            x);
}

__global__ void KernelUpdateAdam(dtype *val, dtype *grad, int row, int col,
        dtype *aux_mean,
        dtype *aux_square,
        const bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        int count_i = i / col;
        if (indexers[count_i]) {
            if (row > 1 && col > 1) {
                grad[i] += val[i] * reg;
            }
            aux_mean[i] = belta1 * aux_mean[i] + (1 - belta1) * grad[i];
            aux_square[i] = belta2 * aux_square[i] + (1 - belta2) * grad[i] *
                grad[i];
            dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2,
                        iters[count_i] + 1)) / (1 - cuda_pow(belta1,
                            iters[count_i] + 1));
            dtype square_plus_eps = aux_square[i] + eps;
            val[i] = val[i] - aux_mean[i] * lr_t / cuda_sqrt(square_plus_eps);
        }
    }
}

__global__ void KernelSelfPlusIters(const bool *indexers, int *iters,
        int count) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count; i += step) {
        if (indexers[i]) {
            ++iters[i];
        }
    }
}

void UpdateAdam(dtype *val, dtype *grad, int row, int col, dtype *aux_mean,
        dtype *aux_square,
        const bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    KernelUpdateAdam<<<block_count, TPB>>>(val, grad, row, col, aux_mean,
            aux_square, indexers, iters, belta1, belta2, alpha, reg, eps);
    block_count = DefaultBlockCount(row);
    KernelSelfPlusIters<<<block_count, TPB>>>(indexers, iters, row);
}

void *GraphHostAlloc() {
    void *m;
    CallCuda(cudaHostAlloc(&m, 10000000, cudaHostAllocWriteCombined));
    if (m == NULL) {
        abort();
    }
    return m;
}

}
