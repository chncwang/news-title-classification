#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Utf.h"
#include "MyLib.h"
#include "Concat.h"
#include "UniOP.h"

class GraphBuilder {
public:
    std::vector<LookupNode> _input_nodes;
    WindowBuilder _window_builder;
    std::vector<UniNode> _uni_nodes;
    MaxPoolNode _max_pool_node;
    LinearNode _neural_output;

    Graph *_graph;
    ModelParams *_modelParams;
    const static int max_sentence_length = 100;

    GraphBuilder() = default;
    GraphBuilder(const GraphBuilder&) = default;
    GraphBuilder(GraphBuilder&&) = default;

    void createNodes(int length_upper_bound) {
        _input_nodes.resize(length_upper_bound);
        _window_builder.resize(length_upper_bound);
        _uni_nodes.resize(length_upper_bound);
    }

    void initial(Graph *pcg, ModelParams &model, HyperParams &opts) {
        _graph = pcg;
        for (LookupNode &n : _input_nodes) {
            n.init(opts.wordDim, opts.dropProb);
            n.setParam(&model.words);
        }

        _window_builder.init(opts.wordDim, opts.wordContext);
        for (UniNode &n : _uni_nodes) {
            n.init(opts.hiddenSize, opts.dropProb);
            n.setParam(&model.hidden);
        }

        _max_pool_node.init(opts.hiddenSize, -1);
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
        _window_builder.forward(_graph, input_node_ptrs);
        for (int i = 0; i < feature.m_title_words.size(); ++i) {
            _uni_nodes.at(i).forward(_graph, &_window_builder._outputs.at(i));
        }

        std::vector<Node*> uni_node_ptrs =
            toPointers<UniNode, Node>(_uni_nodes,
                feature.m_title_words.size());
        _max_pool_node.forward(_graph, uni_node_ptrs);
        _neural_output.forward(_graph, &_max_pool_node);
    }
};


#endif /* SRC_ComputionGraph_H_ */
