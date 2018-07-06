#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Utf.h"
#include "MyLib.h"
#include "UniOP.h"
#include <array>

class WordCounter {
public:
    int counter = 0;
    static WordCounter& GetInstance() {
        static WordCounter *w = NULL;
        if (w == NULL) {
            w = new WordCounter;
        }
        return *w;
    }
};

class GraphBuilder {
public:
    LookupNode _input_node;
    UniNode _neural_output;

    Graph *_graph;
    ModelParams *_modelParams;
    const static int max_sentence_length = 100;

    GraphBuilder() = default;
    GraphBuilder(const GraphBuilder&) = default;
    GraphBuilder(GraphBuilder&&) = default;

    void createNodes(int length_upper_bound) {
    }

    void initial(Graph *pcg, ModelParams &model, HyperParams &opts) {
        _graph = pcg;
        _input_node.init(opts.wordDim, opts.dropProb);
        _input_node.setParam(&model.words);

        _neural_output.init(opts.labelSize, -1);
        _neural_output.setParam(&model.olayer_linear);

        _modelParams = &model;
    }

    inline void forward(const Feature &feature, bool bTrain = false) {
        _graph->train = bTrain;
        const std::string &word = feature.m_title_words.at(0);
        _input_node.forward(_graph, word);

        _neural_output.forward(_graph, &_input_node);
    }
};


#endif /* SRC_ComputionGraph_H_ */
