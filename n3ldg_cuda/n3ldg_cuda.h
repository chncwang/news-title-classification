#ifndef N3LDG_CUDA_N3LDG_CUDA_H
#define N3LDG_CUDA_N3LDG_CUDA_H

#include "Def.h"

#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>
#include <cmath>

namespace n3ldg_cuda {

struct NumberPointerArray {
    dtype **value = NULL;
    int len = 0;

    NumberPointerArray() = default;
    NumberPointerArray(NumberPointerArray&&) = default;
    NumberPointerArray(const NumberPointerArray &) {
        abort();
    }
    void init(dtype **host_arr, int len);
    ~NumberPointerArray();
};

struct NumberPointerPointerArray {
    dtype ***value = NULL;
    int len = 0;

    NumberPointerPointerArray() = default;
    NumberPointerPointerArray(NumberPointerPointerArray&&) = default;
    NumberPointerPointerArray(const NumberPointerPointerArray &) {
        abort();
    }
    void init(dtype ***host_arr, int len);
    ~NumberPointerPointerArray();
};

struct NumberArray {
    dtype *value = NULL;
    int len = 0;

    NumberArray() = default;
    NumberArray(NumberArray&&) = default;
    NumberArray(const NumberArray &) = delete;
    void init(dtype *host_arr, int len);
    ~NumberArray();
};

struct IntPointerArray {
    int **value = NULL;
    int len = 0;

    IntPointerArray() = default;
    IntPointerArray(IntPointerArray&&) = default;
    IntPointerArray(const IntPointerArray &) = delete;
    void init(int **host_arr, int len);
    ~IntPointerArray();
};

struct IntArray {
    int *value = NULL;
    int len = 0;

    IntArray() = default;
    IntArray(IntArray&&) = default;
    IntArray(const IntArray &) {
        abort();
    }
    void init(int *host_arr, int len);
    void init(int len);
    ~IntArray();
};

struct BoolArray {
    bool *value = NULL;
    int len = 0;

    BoolArray() = default;
    BoolArray(BoolArray&&) = default;
    BoolArray(const BoolArray &) {
        abort();
    }
    void init(bool *host_arr, int len);
    void copyFromHost(bool *host_arr);
    ~BoolArray();
};

bool Verify(dtype *host, dtype* device, int len, const char* message);

bool Verify(bool *host, bool *device, int len, const char* message);

struct Tensor1D {
    dtype *value = NULL;
    dtype *v = NULL;
    int dim = 0;

    Tensor1D() = default;
    Tensor1D(const Tensor1D &);
    Tensor1D(Tensor1D &&) = default;
    void init(int len);
    void initOnDevice(int len);
    ~Tensor1D();

    void save(std::ofstream &s) const {
    }

    void load(std::ifstream &s) {
    }

    const Mat mat() const {
        return Mat(v, dim, 1);
    }

    Mat mat() {
        return Mat(v, dim, 1);
    }

    const Mat tmat() const {
        return Mat(v, 1, dim);
    }

    void zero() {
        assert(v != NULL);
        memset((void*)v, 0, dim * sizeof(dtype));;
    }

    Mat tmat() {
        return Mat(v, 1, dim);
    }

    const Vec vec() const {
        return Vec(v, dim);
    }

    Vec vec() {
        return Vec(v, dim);
    }

    inline dtype& operator[](const int i) {
        return v[i];  // no boundary check?
    }

    inline const dtype& operator[](const int i) const {
        return v[i];  // no boundary check?
    }

    inline Tensor1D& operator=(const dtype &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a;
        return *this;
    }

    inline Tensor1D& operator=(const std::vector<dtype> &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a[i];
        return *this;
    }

    inline Tensor1D& operator=(const nr::NRVec<dtype> &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a[i];
        return *this;
    }

    inline Tensor1D& operator=(const Tensor1D &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a[i];
        return *this;
    }

    inline void random(dtype bound) {
        dtype min = -bound, max = bound;
        for (int i = 0; i < dim; i++) {
            v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
        }
    }

    bool verify(const char *message) {
        return Verify(v, value, dim, message);
    }

    void copyFromHostToDevice();
    void copyFromDeviceToHost();
};

struct Tensor2D {
    dtype *value = NULL;
    dtype *v = NULL;
    int row = 0;
    int col = 0;
    int size = 0;

    Tensor2D() = default;
    Tensor2D(const Tensor2D &);
    Tensor2D(Tensor2D &&) = default;
    void init(int row, int col);
    void initOnDevice(int row, int col);
    ~Tensor2D();

    void save(std::ofstream &s) const {
    }

    void load(std::ifstream &s) {
    }

    // for embeddings only, embedding matrix: vocabulary  * dim
    // each word's embedding is notmalized
    inline void norm2one(dtype norm = 1.0) {
        dtype sum;
        for (int idx = 0; idx < row; idx++) {
            sum = 0.000001;
            for (int idy = 0; idy < col; idy++) {
                sum += (*this)[idx][idy] * (*this)[idx][idy];
            }
            dtype scale = sqrt(norm / sum);
            for (int idy = 0; idy < col; idy++) {
                (*this)[idx][idy] *= scale;
            }
        }
    }

    void zero() {
        assert(v != NULL);
        memset((void*)v, 0, row * col * sizeof(dtype));;
    }

    const Mat mat() const {
        return Mat(v, row, col);
    }

    Mat mat() {
        return Mat(v, row, col);
    }

    const Vec vec() const {
        return Vec(v, size);
    }

    Vec vec() {
        return Vec(v, size);
    }


    //use it carefully, first col, then row, because rows are allocated successively
    dtype* operator[](const int irow) {
        return &(v[irow*col]);  // no boundary check?
    }

    const dtype* operator[](const int irow) const {
        return &(v[irow*col]);  // no boundary check?
    }

    //use it carefully
    Tensor2D& operator=(const dtype &a) { // assign a to every element
        for (int i = 0; i < size; i++)
            v[i] = a;
        return *this;
    }

    Tensor2D& operator=(const std::vector<dtype> &a) { // assign a to every element
        for (int i = 0; i < size; i++)
            v[i] = a[i];
        return *this;
    }

    Tensor2D& operator=(const std::vector<std::vector<dtype> > &a) { // assign a to every element
        int offset = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                v[offset] = a[i][j];
                offset++;
            }
        }
        return *this;
    }

    Tensor2D& operator=(const nr::NRMat<dtype> &a) { // assign a to every element
        int offset = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                v[offset] = a[i][j];
                offset++;
            }
        }
        return *this;
    }

    Tensor2D& operator=(const Tensor2D &a) { // assign a to every element
        for (int i = 0; i < size; i++)
            v[i] = a.v[i];
        return *this;
    }

    void random(dtype bound) {
        dtype min = -bound, max = bound;
        for (int i = 0; i < size; i++) {
            v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
        }
    }

    // for embeddings only, embedding matrix: dim  * vocabulary
    // each word's embedding is notmalized
    void norm2one() {
        dtype sum;
        for (int idx = 0; idx < col; idx++) {
            sum = 0.000001;
            for (int idy = 0; idy < row; idy++) {
                sum += (*this)[idx][idy] * (*this)[idx][idy];
            }
            dtype scale = sqrt(sum);
            for (int idy = 0; idy < row; idy++) {
                (*this)[idx][idy] /= scale;
            }
        }
    }

    bool verify(const char* message) {
        return Verify(v, value, size, message);
    }

    void copyFromHostToDevice();
    void copyFromDeviceToHost();
};

void Assert(bool v);
void Memset(dtype *p, int len, dtype value);
void Memset(bool *p, int len, bool value);

void UpdateAdam(Tensor2D &val, Tensor2D &grad, Tensor2D &aux_mean,
        Tensor2D &aux_square,
        int &iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps);
void RescaleGrads(std::vector<dtype *> &grads, const std::vector<int> &lens,
        dtype max_scale);

void InitCuda();
void EndCuda();
void CopyFromOneVectorToMultiVectors(const dtype *src, dtype *dest, int count,
        int len);
void Tanh(const dtype *src, const std::vector<dtype*>& dest, dtype* dest2,
        int len, bool is_being_trained, dtype drop_factor,
        const dtype *drop_mask);
NumberPointerArray ToNumberPointerArray(const std::vector<dtype*> &vec);
void CopyForUniNodeForward(const std::vector<dtype*> &xs, const dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len);
void MatrixMultiplyMatrix(dtype *W, dtype *x, dtype *y, int row, int col,
        int count,
        bool useb,
        bool should_x_transpose = false,
        bool should_W_transpose = false);


void CalculateLtyForUniBackward(const std::vector<dtype*> &ly, const dtype *ty,
        const dtype *y,
        const dtype *drop_mask,
        dtype drop_factor,
        dtype *lty,
        int count,
        int dim);
void AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(const dtype *lty,
        const dtype *lx, dtype *b, std::vector<dtype*> &losses, int count,
        int out_dim, int in_dim);
void CalculateDropoutMask(dtype dropout_ratio, int count, int dim,
        dtype *mask);
void ConcatForward(const std::vector<dtype**> &ins, const int *in_offsets,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &outs,
        int count,
        int in_count,
        int out_dim);
void ConcatBackward(const std::vector<dtype*> &out_losses,
        const int *in_offsets,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype**> in_losses,
        int count,
        int in_count,
        int out_dim);
IntPointerArray ToIntPointerArray(const std::vector<int*> &vec);
IntArray ToIntArray(const std::vector<int> &vec);
void LookupForward(const std::vector<int> &xids, const dtype *vocabulary,
        const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int dim,
        std::vector<dtype*> &vals);
void LookupBackward(const std::vector<int> &xids, int unknown_id,
        bool fine_tune,
        const std::vector<dtype*> &losses,
        const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int dim,
        dtype *grad,
        bool *indexers);
void MaxPoolForward(const std::vector<dtype**> &ins, int count,
        const std::vector<int> &in_counts,
        int dim,
        int *hit_inputs,
        std::vector<dtype*> &outs);
void MaxPoolBackward(const std::vector<dtype*> &losses, const int *hit_inputs,
        int count, 
        int dim,
        std::vector<dtype**> &in_losses);
}

#endif
