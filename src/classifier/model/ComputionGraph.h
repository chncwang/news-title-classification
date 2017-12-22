#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Utf.h"
#include "LSTM1.h"
#include "MyLib.h"
#include "Concat.h"
#include "BiOP.h"

class GraphBuilder {
public:
    std::vector<LookupNode> _input_nodes;
    LSTM1Builder _left_to_right_lstm;
    LSTM1Builder _right_to_left_lstm;
    std::vector<BiNode> _bi_nodes;
    MaxPoolNode _max_pool_node;
    LinearNode _neural_output;

    Graph *_graph;
    ModelParams *_modelParams;
    const static int max_sentence_length = 1024;

public:
    void createNodes(int length_upper_bound) {
        _input_nodes.resize(length_upper_bound);
        _bi_nodes.resize(length_upper_bound);
        _left_to_right_lstm.resize(length_upper_bound);
        _right_to_left_lstm.resize(length_upper_bound);
        _max_pool_node.setParam(length_upper_bound);
    }

public:
    void initial(Graph *pcg, ModelParams &model, HyperParams &opts) {
        _graph = pcg;
        for (LookupNode &n : _input_nodes) {
            n.init(opts.wordDim, opts.dropProb);
            n.setParam(&model.words);
        }
        _left_to_right_lstm.init(&model.left_to_right_lstm_param,
                opts.dropProb, true);
        _right_to_left_lstm.init(&model.right_to_left_lstm_param,
                opts.dropProb, false);
        for (BiNode &n : _bi_nodes) {
            n.init(opts.hiddenSize, opts.dropProb);
            n.setParam(&model.bi_param);
            n.activate = frelu;
            n.derivate = drelu;
        }
        _max_pool_node.init(opts.hiddenSize, -1);
        _neural_output.init(opts.labelSize, -1);
        _neural_output.setParam(&model.olayer_linear);
        _modelParams = &model;
    }

public:
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

        for (int i = 0; i<feature.m_title_words.size(); ++i) {
            _bi_nodes.at(i).forward(_graph,
                    &_left_to_right_lstm._hiddens.at(i),
                    &_right_to_left_lstm._hiddens.at(i));
        }
        std::vector<Node*> bi_node_ptrs = toPointers<BiNode, Node>(_bi_nodes,
                feature.m_title_words.size());
        _max_pool_node.forward(_graph, bi_node_ptrs);
        _neural_output.forward(_graph, &_max_pool_node);
    }
};


#endif /* SRC_ComputionGraph_H_ */
