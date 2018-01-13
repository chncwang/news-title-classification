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
#include <curand.h>
#include <curand_kernel.h>
#include "profiler.h"
#include "cnmem.h"
#include <string>

namespace n3ldg_cuda {

using std::cout;
using std::endl;

#if USE_FLOAT
#define cuda_sqrt(x) sqrtf(x)
#define cuda_pow(x, y) powf(x, y)
#define cuda_tanh(x) tanhf(x)
#else
#define cuda_sqrt(x) sqrt(x)
#define cuda_pow(x, y) pow(x, y)
#define cuda_tanh(x) tanh(x)
#endif

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

__global__ void PrintNums(dtype* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%f\n", p[i]);
    }
}


void InitCuda() {
    cudaSetDevice(1);

    cnmemDevice_t device;
    device.size = 10000000000;
    device.device = 1;
    //cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);

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

void CopyFromOneVectorToMultiVectors(const dtype *src, dtype *dest, int count,
        int len) {
    KernelCopyFromOneVectorToMultiVectors<<<
        (len * count - 1 + THREAD_COUNT_PER_BLOCK) / THREAD_COUNT_PER_BLOCK,
    THREAD_COUNT_PER_BLOCK>>>(src, dest, count, len);
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
        if (DeviceAbs(loss) > 0.01) {
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
    char *m;
    CallCuda(MemoryPool::Ins().Malloc((void**)&m,
                (strlen(message) + 1) * sizeof(char)));
    CallCuda(cudaMemcpy(m, message,
                (strlen(message) + 1) * sizeof(char), cudaMemcpyHostToDevice));
    bool success = true;
    bool *dev_success;
    CallCuda(MemoryPool::Ins().Malloc((void**)&dev_success, sizeof(bool)));
    CallCuda(cudaMemcpy(dev_success, &success, sizeof(bool),
                cudaMemcpyHostToDevice));
    KernelVerify<<<block_count, THREAD_COUNT_PER_BLOCK>>>(arr.value, device,
            len, m, dev_success);
    CallCuda(cudaMemcpy(&success, dev_success, sizeof(bool),
                cudaMemcpyDeviceToHost));
//    if (!success) {
//        printf("host:\n");
//        PrintNums<<<1, 1>>>(arr.value, len);
//        cudaDeviceSynchronize();
//        printf("device:\n");
//        PrintNums<<<1, 1>>>(device, len);
//        cudaDeviceSynchronize();
//    }
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    return success;
}

cudaError_t MemoryPool::Malloc(void **p, int size) {
    //CallCnmem(cnmemMalloc(p, size, NULL));
    //return cudaSuccess;

//    return cudaMalloc(p, size);

    //std::cout << "free size:" << free_blocks_.size() << " busy size:" <<
    //    busy_blocks_.size() << std::endl;
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("malloc");
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
        assert(status == cudaSuccess);
        MemoryBlock block(*p, size);
        busy_blocks_.push_back(block);
    }

    profiler.EndEvent();
    return status;
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
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("free");
//    CallCnmem(cnmemFree(p, NULL));

//    return cudaFree(p);

    for (auto it = busy_blocks_.end() - 1; it != busy_blocks_.begin() - 1; --it) {
        if (p == it->p) {
            free_blocks_.push_back(*it);
            busy_blocks_.erase(it);
            break;
        }
    }
    profiler.EndEvent();

    return cudaSuccess;
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
    //if (count >= THREAD_COUNT_PER_BLOCK) {
    //    KernelPrintLine("count_i:%d", count_i);
    //}
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
                b[dim_i] += sum;
            }
        }
    } else {
        if (count_i < count) {
            dim_i -= out_dim;
            int lx_index = dim_i * count + count_i;
            losses[count_i][dim_i] += lx[lx_index];
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

}
