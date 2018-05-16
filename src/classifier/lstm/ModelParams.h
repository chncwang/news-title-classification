#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"
#include "MySoftMaxLoss.h"
#include "LSTM1.h"
#include "BiOP.h"
#include <array>

constexpr int CNN_LAYER = 1;

class ModelParams{
public:
    LookupTable words;
    Alphabet wordAlpha;
    LSTM1Params left_to_right_lstm;
    LSTM1Params right_to_left_lstm;
    BiParams bi_params;

    UniParams olayer_linear;
    MySoftMaxLoss loss;

    bool initial(HyperParams& opts){
        if (words.nVSize <= 0){
            std::cout << "ModelParam initial - words.nVSize:" << words.nVSize << std::endl;
            abort();
        }
        opts.wordDim = words.nDim;
        opts.labelSize = 32;

        left_to_right_lstm.initial(opts.hiddenSize, opts.wordDim);
        right_to_left_lstm.initial(opts.hiddenSize, opts.wordDim);
        bi_params.initial(opts.hiddenSize, opts.hiddenSize, opts.hiddenSize, true);

        olayer_linear.initial(opts.labelSize, opts.hiddenSize * 2, false);
        return true;
    }

    void exportModelParams(ModelUpdate& ada){
        words.exportAdaParams(ada);
        left_to_right_lstm.exportAdaParams(ada);
        right_to_left_lstm.exportAdaParams(ada);
        bi_params.exportAdaParams(ada);
        olayer_linear.exportAdaParams(ada);
    }

    void exportCheckGradParams(CheckGrad& checkgrad){
        checkgrad.add(&olayer_linear.W, "output layer W");
        checkgrad.add(&left_to_right_lstm.cell.W1, "cell W1");
        checkgrad.add(&left_to_right_lstm.cell.W2, "cell W2");
        checkgrad.add(&left_to_right_lstm.cell.b, "cell b");
        checkgrad.add(&left_to_right_lstm.forget.W1, "forget W1");
        checkgrad.add(&left_to_right_lstm.forget.W2, "forget W2");
        checkgrad.add(&left_to_right_lstm.forget.b, "forget b");
        checkgrad.add(&left_to_right_lstm.input.W1, "input W1");
        checkgrad.add(&left_to_right_lstm.input.W2, "input W2");
        checkgrad.add(&left_to_right_lstm.input.b, "input b");
        checkgrad.add(&left_to_right_lstm.output.W1, "output W1");
        checkgrad.add(&left_to_right_lstm.output.W2, "output W2");
        checkgrad.add(&left_to_right_lstm.output.b, "output b");
    }

    void saveModel(std::ofstream &os) {
    }

    void loadModel(std::ifstream &is) {
    }
};

#endif
