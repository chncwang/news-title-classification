#ifndef N3LDG_CUDA_N3LDG_CUDA_H
#define N3LDG_CUDA_N3LDG_CUDA_H

#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace n3ldg_cuda {

struct NumberPointerArray {
    dtype **value = NULL;
    int len = 0;

    NumberPointerArray(const NumberPointerArray &) = delete;
    void init(dtype **host_arr, int len);
    ~NumberPointerArray();
};

struct IntPointerArray {
    dtype **value = NULL;
    int len = 0;

    IntPointerArray(const IntPointerArray &) = delete;
    void init(dtype **host_arr, int len);
    ~IntPointerArray();
};

struct IntArray {
    int *value = NULL;
    int size = 0;

    IntArray(const IntArray &) = delete;
    void init(dtype **host_arr, int len);
    ~IntArray();
};

struct Tensor1D {
    dtype *value = NULL;
    dtype *v = NULL;
    int dim = 0;

    Tensor1D() = default;
    Tensor1D(const Tensor1D &);
    Tensor1D(Tensor1D &&) = default;
    void init(int len);
    ~Tensor1D();

    inline void zero() {
        if(v)memset((void*)v, 0, dim * sizeof(dtype));;
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

    inline Tensor1D& operator=(const vector<dtype> &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a[i];
        return *this;
    }

    inline Tensor1D& operator=(const NRVec<dtype> &a) { // assign a to every element
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
};

struct Tensor2D {
    dtype *value = NULL;
    dtype *v = NULL;
    int row = 0;
    int col = 0;

    Tensor2D() = default;
    Tensor2D(const Tensor2D &);
    Tensor2D(Tensor2D &&) = default;
    void init(int row, int col);
    ~Tensor2D();

    int size() const {
        return row * col;
    }

    void zero() {
        if(v)memset((void*)v, 0, row * col * sizeof(dtype));;
    }

    const Mat mat() const {
        return Mat(v, row, col);
    }

    Mat mat() {
        return Mat(v, row, col);
    }

    const Vec vec() const {
        return Vec(v, size());
    }

    Vec vec() {
        return Vec(v, size());
    }


    //use it carefully, first col, then row, because rows are allocated successively
    dtype* operator[](const int icol) {
        return &(v[icol*row]);  // no boundary check?
    }

    const dtype* operator[](const int icol) const {
        return &(v[icol*row]);  // no boundary check?
    }

    //use it carefully
    Tensor2D& operator=(const dtype &a) { // assign a to every element
        for (int i = 0; i < size(); i++)
            v[i] = a;
        return *this;
    }

    Tensor2D& operator=(const vector<dtype> &a) { // assign a to every element
        for (int i = 0; i < size(); i++)
            v[i] = a[i];
        return *this;
    }

    Tensor2D& operator=(const vector<vector<dtype> > &a) { // assign a to every element
        int offset = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                v[offset] = a[i][j];
                offset++;
            }
        }
        return *this;
    }

    Tensor2D& operator=(const NRMat<dtype> &a) { // assign a to every element
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
        for (int i = 0; i < size(); i++)
            v[i] = a.v[i];
        return *this;
    }

    void random(dtype bound) {
        dtype min = -bound, max = bound;
        for (int i = 0; i < size(); i++) {
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
};

void UpdateAdam(Tensor2D &val, Tensor2D &grad, Tensor2D &aux_mean,
        Tensor2D &aux_square, int &iter, dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps);
}

#endif
