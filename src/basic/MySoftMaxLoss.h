#ifndef MY_SOFTMAXLOSS_H_
#define MY_SOFTMAXLOSS_H_

#include "MyLib.h"
#include "Metric.h"
#include "Node.h"
#include "Category.h"
#if USE_GPU
#include "n3ldg_cuda.h"
#endif

class MySoftMaxLoss {
public:
#if USE_GPU
    dtype loss(const std::vector<PNode> &x, const std::vector<int> &answers,
            n3ldg_cuda::DeviceInt &correct,
            int batchsize = 1) {
        std::vector<dtype*> vals, losses;
        vals.reserve(x.size());
        losses.reserve(x.size());
        for (PNode n : x) {
            vals.push_back(n->val.value);
            losses.push_back(n->loss.value);
        }
        n3ldg_cuda::SoftMaxLoss(vals, losses, correct.value, answers,
                batchsize, x.size(), x.at(0)->dim);
        return -1.0f;
    }
#endif
    inline dtype loss(PNode x, Category answer, Metric& metric, int batchsize = 1) {
        int nDim = x->dim;
        int labelsize = 32;
        if (labelsize != nDim) {
            std::cerr << "softmax_loss error: dim size invalid" << std::endl;
            abort();
        }

        NRVec<dtype> scores(nDim);

        dtype cost = 0.0;
        int optLabel = -1;
        for (int i = 0; i < nDim; ++i) {
            if (optLabel < 0 || x->val[i] > x->val[optLabel])
                optLabel = i;
        }

        dtype sum1 = 0, sum2 = 0, maxScore = x->val[optLabel];
        for (int i = 0; i < nDim; ++i) {
            scores[i] = -1e10;
            scores[i] = exp(x->val[i] - maxScore);
            if (answer == i)
                sum1 += scores[i];
            sum2 += scores[i];
        }
        cost += (log(sum2) - log(sum1)) / batchsize;
        if (answer == optLabel) {
            metric.correct_label_count++;
        }
        metric.overall_label_count++;

        for (int i = 0; i < nDim; ++i) {
            float t = answer == i ? 1.0 : 0.0;
            x->loss[i] = (scores[i] / sum2 - t) / batchsize;
        }

        return cost;
    }

    inline dtype predict(PNode x, int& y, int excluded_class) {
        n3ldg_assert(excluded_class >= -1 && excluded_class < 3, "excluded class is " << excluded_class);
        int nDim = x->dim;

        int optLabel = -1;
        for (int i = 0; i < nDim; ++i) {
            if (optLabel < 0 || x->val[i] >  x->val[optLabel]) {
                if (i != excluded_class) {
                    optLabel = i;
                }
            }
        }
        y = optLabel;

        dtype prob = 0.0;
        dtype sum = 0.0;
        NRVec<dtype> scores(nDim);
        dtype maxScore = x->val[optLabel];
        for (int i = 0; i < nDim; ++i) {
            scores[i] = exp(x->val[i] - maxScore);
            sum += scores[i];
        }
        prob = scores[optLabel] / sum;
        return prob;
    }

    inline dtype cost(PNode x, Category answer, int batchsize = 1) {
        int nDim = x->dim;
        int labelsize = 3;
        if (labelsize != nDim) {
            std::cerr << "softmax_loss error: dim size invalid" << std::endl;
            return -1.0;
        }

        NRVec<dtype> scores(nDim);

        dtype cost = 0.0;

        int optLabel = -1;
        for (int i = 0; i < nDim; ++i) {
            if (optLabel < 0 || x->val[i] > x->val[optLabel])
                optLabel = i;
        }
        dtype sum1 = 0, sum2 = 0, maxScore = x->val[optLabel];
        for (int i = 0; i < nDim; ++i) {
            scores[i] = -1e10;
            scores[i] = exp(x->val[i] - maxScore);
            if (answer == i)
                sum1 += scores[i];
            sum2 += scores[i];
        }
        cost += (log(sum2) - log(sum1)) / batchsize;
        return cost;
    }

};


#endif /* _SOFTMAXLOSS_H_ */
