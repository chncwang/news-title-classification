#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"
#include <array>
#include <iostream>

constexpr int CNN_LAYER = 1;

class ModelParams{
public:
    LookupTable words;
    Alphabet wordAlpha;

    UniParams olayer_linear;
    SoftMaxLoss loss;

    bool initial(HyperParams& opts){
        if (words.nVSize <= 0){
            std::cout << "ModelParam initial - words.nVSize:" << words.nVSize << std::endl;
            abort();
        }
        opts.wordDim = words.nDim;
        opts.labelSize = 32;
        olayer_linear.initial(opts.labelSize, opts.wordDim, true);
        return true;
    }

    void exportModelParams(ModelUpdate& ada){
        words.exportAdaParams(ada);
        olayer_linear.exportAdaParams(ada);
    }

    void exportCheckGradParams(CheckGrad& checkgrad){
        checkgrad.add(&words.E, "words E");
        checkgrad.add(&olayer_linear.b, "output layer b");
        checkgrad.add(&olayer_linear.W, "output layer W");
    }

    void saveModel(std::ofstream &os) {
        wordAlpha.write(os);
        words.save(os);
        olayer_linear.save(os);
    }

    void loadModel(std::ifstream &is) {
        wordAlpha.read(is);
        words.load(is, &wordAlpha);
        olayer_linear.load(is);
    }
};

#endif
