#ifndef N3LDG_CUDA_MEMORY_POOL_H
#define N3LDG_CUDA_MEMORY_POOL_H

#include <vector>
#include <list>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace n3ldg_cuda {

struct MemoryBlock {
    void *p;
    int size;

    MemoryBlock(void *p, int size) {
        this->p = p;
        this->size = size;
    }
};

class MemoryPool {
public:
    MemoryPool(const MemoryPool &) = delete;
    static MemoryPool& Ins() {
        static MemoryPool *p;
        if (p == NULL) {
            p = new MemoryPool;
        }
        return *p;
    }

    cudaError_t Malloc(void **p, int size);

    cudaError_t Free(void *p);

    void FreePool();
private:
    MemoryPool() = default;
    std::vector<MemoryBlock> free_blocks_;
    std::vector<MemoryBlock> busy_blocks_;
};

}

#endif
