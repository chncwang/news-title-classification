#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"
#include "MySoftMaxLoss.h"

class ModelParams{
public:
    LookupTable words;
    Alphabet wordAlpha;
    UniParams hidden;

    UniParams olayer_linear;
    MySoftMaxLoss loss;

    bool initial(HyperParams& opts){
        if (words.nVSize <= 0){
            std::cout << "ModelParam initial - words.nVSize:" << words.nVSize << std::endl;
            abort();
        }
        opts.wordDim = words.nDim;
        opts.labelSize = 32;
        hidden.initial(opts.hiddenSize, (1 + 2 * opts.wordContext) * opts.wordDim, true);
        olayer_linear.initial(opts.labelSize, opts.hiddenSize, true);
        return true;
    }

    void exportModelParams(ModelUpdate& ada){
        words.exportAdaParams(ada);
        hidden.exportAdaParams(ada);
        olayer_linear.exportAdaParams(ada);
    }

    void exportCheckGradParams(CheckGrad& checkgrad){
        checkgrad.add(&words.E, "words E");
        //checkgrad.add(&hidden_linear.W, "hidden w");
        //checkgrad.add(&hidden_linear.b, "hidden b");
        checkgrad.add(&olayer_linear.b, "output layer W");
        checkgrad.add(&olayer_linear.W, "output layer W");
    }
};

#endif
