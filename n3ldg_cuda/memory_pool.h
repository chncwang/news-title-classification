#ifndef N3LDG_CUDA_MEMORY_POOL_H
#define N3LDG_CUDA_MEMORY_POOL_H

#include <vector>
#include <list>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace n3ldg_cuda {

struct MemoryBlock {
    void *p;
    int64_t size;

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
            p->busy_blocks_.reserve(10000);
        }
        return *p;
    }

    cudaError_t Malloc(void **p, int size);
    cudaError_t Free(void *p);
private:
    MemoryPool() = default;
    std::list<MemoryBlock> free_blocks_;
    std::unordered_map<void *, MemoryBlock> busy_blocks_;
};

}

#endif
