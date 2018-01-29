/*dIndexers.verify("SparseParam clearGrad")*
* Driver.h
*
*  Created on: Mar 18, 2015
*      Author: mszhang
*/

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"
#include "Category.h"
#include "MySoftMaxLoss.h"
#include "Targets.h"
#include "profiler.h"

//A native neural network classfier using only word embeddings

class Driver {
public:
    Driver(int memsize) {}

    ~Driver() {}

public:
    Graph _cg;  // build neural graphs
    vector<GraphBuilder> _builders;
    ModelParams _modelparams;  // model parameters
    HyperParams _hyperparams;

    Metric _metric;
    CheckGrad _checkgrad;
    ModelUpdate _ada;  // model update

public:
    //embeddings are initialized before this separately.
    void initial() {
        if (!_hyperparams.bValid()) {
            std::cout << "hyper parameter initialization Error, Please check!"
                << std::endl;
            abort();
        }
        if (!_modelparams.initial(_hyperparams)) {
            std::cout << "model parameter initialization Error, Please check!"
                << std::endl;
            abort();
        }
        _modelparams.exportModelParams(_ada);
#if USE_GPU
//        for (BaseParam *p : _ada._params) {
//            p->copyFromDeviceToHost();
//        }
#endif
        _modelparams.exportCheckGradParams(_checkgrad);
        _builders.resize(_hyperparams.batch);

        for (int idx = 0; idx < _hyperparams.batch; idx++) {
            _builders[idx].createNodes(GraphBuilder::max_sentence_length);
            _builders[idx].initial(&_cg, _modelparams, _hyperparams);
        }

        setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha,
            _hyperparams.adaEps);
    }

#if USE_GPU
    void clearValueOnDevice() {
//        std::vector<Node*> inputs, concats, buckets, unis, max_pools, linears;
//        for (GraphBuilder &bu : _builders) {
//            std::vector<Node*> inputs_t =
//                toPointers<LookupNode, Node>(bu._input_nodes);
//            for (Node *p : inputs_t) {
//                inputs.push_back(p);
//            }

//            std::vector<Node *> concat_t = toPointers<ConcatNode,
//                Node>(bu._window_builder._outputs);
//            for (Node *p : concat_t) {
//                concats.push_back(p);
//            }

//            buckets.push_back(&bu._window_builder._bucket);

//            for (const std::vector<UniNode> &unis : bu._uni_nodes) {
//                std::vector<Node *> unis_t = toPointers<UniNode,
//                    Node>(unis);
//                for (Node *p: unis_t) {
//                    unis.push_back(p);
//                }
//            }

//            max_pools.push_back(&bu._max_pool_node);
//            linears.push_back(&bu._neural_output);
//        }
//        clearNodes(inputs, _hyperparams.wordDim);
//        clearNodes(max_pools, _hyperparams.hiddenSize);
//        clearNodes(concats, (1 + 2 * _hyperparams.wordContext) *
//                _hyperparams.wordDim);
//        clearNodes(buckets, _hyperparams.wordDim);
//        clearNodes(unis, _hyperparams.hiddenSize);
    }
#endif

    inline dtype train(const vector<Example> &examples, int iter) {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent("train");
        resetEval();
        _cg.clearValue();
        int example_num = examples.size();
        if (example_num > _builders.size()) {
            std::cout << "Driver train - input example number larger than predefined batch number example_num:" << example_num
                << " _builders.size():" << _builders.size() << std::endl;
            abort();
        }

        dtype cost = 0.0;

        profiler.BeginEvent("build graph");
        for (int count = 0; count < example_num; count++) {
            const Example &example = examples[count];

            //forward
            _builders[count].forward(example.m_feature, true);

        }
        profiler.EndEvent();
        _cg.compute();
#if USE_GPU
        std::vector<Node*> outputs;
        outputs.reserve(example_num);
        std::vector<int> answers;
        answers.reserve(example_num);
        for (int count = 0; count < example_num; count++) {
            const Example &example = examples[count];
            answers.push_back(static_cast<int>(example.m_category));
            outputs.push_back(&_builders[count]._neural_output);
        }
        n3ldg_cuda::DeviceInt correct_count;
        correct_count.init();
        profiler.BeginEvent("softmax");
        _modelparams.loss.loss(outputs, answers, correct_count, example_num);
#if USE_GPU
        profiler.EndCudaEvent();
#else
        profiler.EndEvent();
#endif
        correct_count.copyFromDeviceToHost();
#if TEST_CUDA
        int previous_correct_count = _metric.correct_label_count;
        for (int count = 0; count < example_num; count++) {
            const Example &example = examples[count];
            cost += _modelparams.loss.loss(&_builders[count]._neural_output,
                example.m_category, _metric, example_num);
        }
        n3ldg_cuda::Assert(correct_count.v == _metric.correct_label_count -
                previous_correct_count);
        for (int count = 0; count < example_num; count++) {
            n3ldg_cuda::Assert(_builders[count]._neural_output.loss.verify(
                        "softmax"));
        }
#endif
        _metric.overall_label_count += example_num;
        _metric.correct_label_count += correct_count.v;
#else
        for (int count = 0; count < example_num; count++) {
            const Example &example = examples[count];
            cost += _modelparams.loss.loss(&_builders[count]._neural_output,
                example.m_category, _metric, example_num);
        }
#endif
        _cg.backward();
#if USE_GPU
        profiler.EndCudaEvent();
#else
        profiler.EndEvent();
#endif
        return cost;
    }

    inline void predict(const Feature &feature, Category &result, int excluded_class) {
        _cg.clearValue();
        _builders[0].forward(feature);
        _cg.compute();

        int intResult;
        _modelparams.loss.predict(&_builders[0]._neural_output, intResult, excluded_class );
        result = static_cast<Category>(intResult);
    }

    inline dtype cost(const Example &example) {
        _cg.clearValue();
        _builders[0].forward(example.m_feature, true);
        _cg.compute();

        dtype cost = _modelparams.loss.cost(&_builders[0]._neural_output,
            example.m_category, 1);

        return cost;
    }


    void updateModel() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent("update model");
        _ada.updateAdam(10);
#if USE_GPU
        profiler.EndCudaEvent();
#else
        profiler.EndEvent();
#endif
    }

    void checkgrad(const vector<Example> &examples, int iter) {
        ostringstream out;
        out << "Iteration: " << iter;
        _checkgrad.check(this, examples, out.str());
    }

private:
    inline void resetEval() {
        _metric.reset();
    }


    inline void
        setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
        _ada._alpha = adaAlpha;
        _ada._eps = adaEps;
        _ada._reg = nnRegular;
    }

};

#endif /* SRC_Driver_H_ */
