#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "ConditionalLSTM.h"
#include "Utf.h"


// Each model consists of two parts, building neural graph and defining output losses.
class GraphBuilder {
public:
    vector<LookupNode> _inputNodes;
    ConditionalLSTMBuilder _left2right;
    ConditionalLSTMBuilder _right2left;
    std::vector<ConcatNode> _concat_nodes;
    MaxPoolNode _max_pooling;
    ReluNode _relu_node;
    LinearNode _neural_output;

    Graph *_graph;
    ModelParams *_modelParams;
    const static int max_sentence_length = 1024;

public:
    //allocate enough nodes
    void createNodes(int length_upper_bound) {
        _inputNodes.resize(length_upper_bound);
        _left2right.resize(length_upper_bound);
        _right2left.resize(length_upper_bound);
        _concat_nodes.resize(length_upper_bound);
        _max_pooling.setParam(length_upper_bound);
    }

public:
    void initial(Graph *pcg, ModelParams &model, HyperParams &opts) {
        _graph = pcg;
        for (LookupNode &n : _inputNodes) {
            n.init(opts.wordDim, opts.dropProb);
            n.setParam(&model.words);
        }
        _left2right.init(opts.dropProb, &model.target_left_to_right_lstm_params, true);
        _right2left.init(opts.dropProb, &model.target_right_to_left_lstm_params, false);

        for (auto &n : _concat_nodes) {
            n.init(opts.hiddenSize * 2, -1);
        }
        _relu_node.init(opts.hiddenSize * 2, -1);
        _max_pooling.init(opts.hiddenSize * 2, -1);
        _neural_output.setParam(&model.olayer_linear);
        _neural_output.init(opts.labelSize, -1);
        _modelParams = &model;
    }

public:
    // some nodes may behave different during training and decode, for example, dropout
    inline void forward(const Feature &feature, bool bTrain = false) {
        _graph->train = bTrain;

        const std::vector<std::string> &target_words = getStanceTargetWords(feature.m_target);

        for (int i = 0; i < target_words.size(); ++i) {
            _inputNodes.at(i).forward(_graph, target_words.at(i));
        }

        for (int i = 0; i < feature.m_tweet_words.size(); ++i) {
            _inputNodes.at(i + target_words.size()).forward(_graph, feature.m_tweet_words.at(i));
        }

        vector<PNode> inputNodes;
        int totalSize = feature.m_tweet_words.size() + target_words.size();
        for (int i = 0; i < totalSize; ++i) {
            inputNodes.push_back(&_inputNodes.at(i));
        }

        _left2right.setParam(&_modelParams->target_left_to_right_lstm_params, &_modelParams->tweet_left_to_right_lstm_params, target_words.size());
        _right2left.setParam(&_modelParams->target_right_to_left_lstm_params, &_modelParams->tweet_right_to_left_lstm_params, target_words.size());

        _left2right.forward(_graph, inputNodes, target_words.size());
        _right2left.forward(_graph, inputNodes, target_words.size());


        for (int i = 0; i < feature.m_tweet_words.size(); ++i) {
            _concat_nodes.at(i).forward(_graph, &_left2right._hiddens.at(target_words.size() + i),
                &_right2left._hiddens.at(target_words.size() + i));
        }
        _max_pooling.forward(_graph, toPointers<ConcatNode, Node>(_concat_nodes, feature.m_tweet_words.size()));
        _relu_node.forward(_graph, &_max_pooling);

        _neural_output.forward(_graph, &_relu_node);
    }
};


#endif /* SRC_ComputionGraph_H_ */
