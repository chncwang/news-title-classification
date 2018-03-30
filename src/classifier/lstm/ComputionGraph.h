#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Utf.h"
#include "MyLib.h"
#include "Concat.h"
#include "UniOP.h"
#include "BiOP.h"
#include "LSTM1.h"
#include <array>

class GraphBuilder {
public:
    std::vector<LookupNode> _input_nodes;
    LSTM1Builder _left_to_right_lstm;
    LSTM1Builder _right_to_left_lstm;
    std::vector<ConcatNode> _concat_nodes;
    MinPoolNode _max_pool_node;
    LinearNode _neural_output;

    Graph *_graph;
    ModelParams *_modelParams;
    const static int max_sentence_length = 100;

    GraphBuilder() = default;
    GraphBuilder(const GraphBuilder&) = default;
    GraphBuilder(GraphBuilder&&) = default;

    void createNodes(int length_upper_bound) {
        _input_nodes.resize(length_upper_bound);
        _left_to_right_lstm.resize(length_upper_bound);
        _right_to_left_lstm.resize(length_upper_bound);
        _concat_nodes.resize(length_upper_bound);
    }

    void initial(Graph *pcg, ModelParams &model, HyperParams &opts) {
        _graph = pcg;
        for (LookupNode &n : _input_nodes) {
            n.init(opts.wordDim, opts.dropProb);
            n.setParam(&model.words);
        }

        _left_to_right_lstm.init(&model.left_to_right_lstm, opts.dropProb,
                true);
        _right_to_left_lstm.init(&model.right_to_left_lstm, opts.dropProb,
                false);
        for (ConcatNode &concat : _concat_nodes) {
            concat.init(opts.hiddenSize * 2, opts.dropProb);
        }

        _max_pool_node.init(opts.hiddenSize * 2, -1);
        _neural_output.init(opts.labelSize, -1);
        _neural_output.setParam(&model.olayer_linear);
        _modelParams = &model;
    }

    inline void forward(const Feature &feature, bool bTrain = false) {
        _graph->train = bTrain;
        for (int i = 0; i < feature.m_title_words.size(); ++i) {
            const std::string &word = feature.m_title_words.at(i);
            _input_nodes.at(i).forward(_graph, word);
        }

        std::vector<Node*> input_node_ptrs =
            toPointers<LookupNode, Node>(_input_nodes,
                    feature.m_title_words.size());
        _left_to_right_lstm.forward(_graph, input_node_ptrs);
        _right_to_left_lstm.forward(_graph, input_node_ptrs);
        for (int i = 0; i < feature.m_title_words.size(); ++i) {
            _concat_nodes.at(i).forward(_graph,
                    &_left_to_right_lstm._hiddens.at(i),
                    &_right_to_left_lstm._hiddens.at(i));
        }

        std::vector<Node*> bi_node_ptrs = toPointers<ConcatNode, Node>(_concat_nodes,
                feature.m_title_words.size());
        _max_pool_node.forward(_graph, bi_node_ptrs);
        _neural_output.forward(_graph, &_max_pool_node);
    }
};


#endif /* SRC_ComputionGraph_H_ */
