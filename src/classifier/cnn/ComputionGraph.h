#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Utf.h"
#include "MyLib.h"
#include "Concat.h"
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
    std::vector<LookupNode> _input_nodes;
    std::array<WindowBuilder, CNN_LAYER> _window_builder;
    std::array<std::vector<UniNode>, CNN_LAYER> _uni_nodes;
    AvgPoolNode _max_pool_node;
    LinearNode _neural_output;

    Graph *_graph;
    ModelParams *_modelParams;
    const static int max_sentence_length = 100;

    GraphBuilder() = default;
    GraphBuilder(const GraphBuilder&) = default;
    GraphBuilder(GraphBuilder&&) = default;

    void createNodes(int length_upper_bound) {
        _input_nodes.resize(length_upper_bound);
        for (int i = 0; i < CNN_LAYER; ++i) {
            _window_builder.at(i).resize(length_upper_bound);
            _uni_nodes.at(i).resize(length_upper_bound);
        }
    }

    void initial(Graph *pcg, ModelParams &model, HyperParams &opts) {
        _graph = pcg;
        for (LookupNode &n : _input_nodes) {
            n.init(opts.wordDim, opts.dropProb);
            n.setParam(&model.words);
        }

        for (int i = 0; i < CNN_LAYER; ++i) {
            _window_builder.at(i).init(i == 0? opts.wordDim : opts.hiddenSize, opts.wordContext);
            for (UniNode &n : _uni_nodes.at(i)) {
                n.init(opts.hiddenSize, opts.dropProb);
                n.setFunctions(ftanh, dtanh);
                n.setParam(&model.hidden.at(i));
            }
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
        WordCounter::GetInstance().counter += feature.m_title_words.size();
        for (int i = 0; i < CNN_LAYER; ++i) {
            if (i == 0) {
                _window_builder.at(i).forward(_graph, input_node_ptrs);
            } else {
                std::vector<Node*> uni_node_ptrs = toPointers<UniNode, Node>(
                        _uni_nodes.at(i - 1), feature.m_title_words.size());
                _window_builder.at(i).forward(_graph, uni_node_ptrs);
            }
            for (int j = 0; j < feature.m_title_words.size(); ++j) {
                _uni_nodes.at(i).at(j).forward(_graph, &_window_builder.at(i)._outputs.at(j));
            }
        }

        std::vector<Node*> uni_node_ptrs =
            toPointers<UniNode, Node>(_uni_nodes.at(CNN_LAYER - 1),
                feature.m_title_words.size());
        _max_pool_node.forward(_graph, uni_node_ptrs);
        _neural_output.forward(_graph, &_max_pool_node);
    }
};


#endif /* SRC_ComputionGraph_H_ */
