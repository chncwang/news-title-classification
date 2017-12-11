#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "ConditionalLSTM.h"
#include "Utf.h"


// Each model consists of two parts, building neural graph and defining output losses.
class GraphBuilder {
public:
    vector<LookupNode> _inputNodes;
    LinearNode _neural_output;

    Graph *_graph;
    ModelParams *_modelParams;
    const static int max_sentence_length = 1024;

public:
    //allocate enough nodes
    void createNodes(int length_upper_bound) {
        _inputNodes.resize(length_upper_bound);
    }

public:
    void initial(Graph *pcg, ModelParams &model, HyperParams &opts) {
        _graph = pcg;
        for (LookupNode &n : _inputNodes) {
            n.init(opts.wordDim, opts.dropProb);
            n.setParam(&model.words);
        }
        _modelParams = &model;
    }

public:
    // some nodes may behave different during training and decode, for example, dropout
    inline void forward(const Feature &feature, bool bTrain = false) {
        _graph->train = bTrain;

//        const std::vector<std::string> &target_words = getStanceTargetWords(feature.m_target);

//        for (int i = 0; i < target_words.size(); ++i) {
//            _inputNodes.at(i).forward(_graph, target_words.at(i));
//        }
    }
};


#endif /* SRC_ComputionGraph_H_ */
