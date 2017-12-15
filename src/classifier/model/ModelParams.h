#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"
#include "MySoftMaxLoss.h"
#include "LSTM1.h"
#include "BiOP.h"

class ModelParams{
public:
    LookupTable words;
    Alphabet wordAlpha;
    LSTM1Params left_to_right_lstm_param;
    LSTM1Params right_to_left_lstm_param;
    BiParams bi_param;

    UniParams olayer_linear;
    MySoftMaxLoss loss;

    bool initial(HyperParams& opts){
        if (words.nVSize <= 0){
            std::cout << "ModelParam initial - words.nVSize:" << words.nVSize << std::endl;
            abort();
        }
        opts.wordDim = words.nDim;
        opts.labelSize = 32;
        left_to_right_lstm_param.initial(opts.hiddenSize, opts.wordDim);
        right_to_left_lstm_param.initial(opts.hiddenSize, opts.wordDim);
        bi_param.initial(opts.hiddenSize, opts.hiddenSize, opts.hiddenSize);
        olayer_linear.initial(opts.labelSize, opts.hiddenSize, true);
        return true;
    }

    void exportModelParams(ModelUpdate& ada){
        words.exportAdaParams(ada);
        left_to_right_lstm_param.exportAdaParams(ada);
        right_to_left_lstm_param.exportAdaParams(ada);
        bi_param.exportAdaParams(ada);
        olayer_linear.exportAdaParams(ada);
    }

    void exportCheckGradParams(CheckGrad& checkgrad){
        checkgrad.add(&words.E, "words E");
        //checkgrad.add(&hidden_linear.W, "hidden w");
        //checkgrad.add(&hidden_linear.b, "hidden b");
        checkgrad.add(&olayer_linear.b, "output layer W");
        checkgrad.add(&olayer_linear.W, "output layer W");
    }

    void saveModel(std::ofstream &os) const{
        wordAlpha.write(os);
        words.save(os);
        olayer_linear.save(os);
    }

    void loadModel(std::ifstream &is){
        wordAlpha.read(is);
        words.load(is, &wordAlpha);
        olayer_linear.load(is);
    }

};

#endif
