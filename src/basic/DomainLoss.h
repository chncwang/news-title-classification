#ifndef STANCE_DECTION_COMP_DOMAIN_LOSS_H
#define STANCE_DECTION_COMP_DOMAIN_LOSS_H

#include <array>
#include "SoftMaxLoss.h"
#include "MyLib.h"

dtype loss(PNode x, Target answer, Metric metric, int batchsize) {
    n3ldg_assert(x->dim == DOMAIN_SIZE, "dim is " << x->dim);
    std::vector<dtype> vector_answer;
    for (int i = 0; i < DOMAIN_SIZE; ++i) {
        vector_answer.push_back(DOMAIN_TARGETRS.at(i) == answer ? 1 : 0);
    }
    return loss(x, vector_answer, metric, batchsize);
}

#endif