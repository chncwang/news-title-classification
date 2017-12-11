#ifndef CONDITIONAL_LSTM`
#define CONDITIONAL_LSTM

#include "MyLib.h"
#include "Node.h"
#include "BiOP.h"
#include "AtomicOP.h"
#include "Graph.h"

struct ConditionalLSTMParams {
    BiParams input;
    BiParams output;
    BiParams forget;
    BiParams cell;

    ConditionalLSTMParams() {}

    inline void exportAdaParams(ModelUpdate& ada) {
        input.exportAdaParams(ada);
        output.exportAdaParams(ada);
        forget.exportAdaParams(ada);
        cell.exportAdaParams(ada);
    }

    inline void initial(int nOSize, int nISize) {
        input.initial(nOSize, nOSize, nISize, true);
        output.initial(nOSize, nOSize, nISize, true);
        forget.initial(nOSize, nOSize, nISize, true);
        cell.initial(nOSize, nOSize, nISize, true);

    }

    inline int inDim() {
        return input.W2.inDim();
    }

    inline int outDim() {
        return input.W2.outDim();
    }

    inline void save(std::ofstream &os) const {
        input.save(os);
        output.save(os);
        forget.save(os);
        cell.save(os);
    }

    inline void load(std::ifstream &is) {
        input.load(is);
        output.load(is);
        forget.load(is);
        cell.load(is);
    }

};

// standard LSTM1 using tanh as activation function
// other conditions are not implemented unless they are clear
class ConditionalLSTMBuilder {
public:
    int _inDim;
    int _outDim;
    vector<BiNode> _inputgates;
    vector<BiNode> _forgetgates;
    vector<BiNode> _halfcells;

    vector<PMultiNode> _inputfilters;
    vector<PMultiNode> _forgetfilters;

    vector<PAddNode> _cells;
    vector<BiNode> _outputgates;
    vector<TanhNode> _halfhiddens;
    vector<PMultiNode> _hiddens;  // intermediate result without dropout

    BucketNode _bucket;

    bool _left2right;

public:
    ConditionalLSTMBuilder() {
        clear();
    }

    ~ConditionalLSTMBuilder() {
        clear();
    }

public:
    void setParam(ConditionalLSTMParams* targetParams, ConditionalLSTMParams *tweetParams, int targetLength) {
        int maxsize = _inputgates.size();
        for (int idx = 0; idx < maxsize; idx++) {
            ConditionalLSTMParams *param = idx < targetLength ? targetParams : tweetParams;
            _inputgates[idx].setParam(&param->input);
            _forgetgates[idx].setParam(&param->forget);
            _outputgates[idx].setParam(&param->output);
            _halfcells[idx].setParam(&param->cell);
            _inputgates[idx].setFunctions(&fsigmoid, &dsigmoid);
            _forgetgates[idx].setFunctions(&fsigmoid, &dsigmoid);
            _outputgates[idx].setFunctions(&fsigmoid, &dsigmoid);
            _halfcells[idx].setFunctions(&ftanh, &dtanh);
        }
    }

    void init(dtype dropout, ConditionalLSTMParams *targetParams, bool left2right = true) {

        _inDim = targetParams->input.W2.inDim();
        _outDim = targetParams->input.W2.outDim();
        _left2right = left2right;
        int maxsize = _inputgates.size();

        for (int idx = 0; idx < maxsize; idx++) {
            _inputgates[idx].init(_outDim, -1);
            _forgetgates[idx].init(_outDim, -1);
            _halfcells[idx].init(_outDim, -1);
            _inputfilters[idx].init(_outDim, -1);
            _forgetfilters[idx].init(_outDim, -1);
            _cells[idx].init(_outDim, -1);
            _outputgates[idx].init(_outDim, -1);
            _halfhiddens[idx].init(_outDim, -1);
            _hiddens[idx].init(_outDim, dropout);
        }

        _bucket.init(_outDim, -1);

    }

    inline void resize(int maxsize) {
        _inputgates.resize(maxsize);
        _forgetgates.resize(maxsize);
        _halfcells.resize(maxsize);
        _inputfilters.resize(maxsize);
        _forgetfilters.resize(maxsize);
        _cells.resize(maxsize);
        _outputgates.resize(maxsize);
        _halfhiddens.resize(maxsize);
        _hiddens.resize(maxsize);
    }

    //whether vectors have been allocated
    inline bool empty() {
        return _hiddens.empty();
    }

    inline void clear() {
        _inputgates.clear();
        _forgetgates.clear();
        _halfcells.clear();
        _inputfilters.clear();
        _forgetfilters.clear();
        _cells.clear();
        _outputgates.clear();
        _halfhiddens.clear();
        _hiddens.clear();

        _left2right = true;
        _inDim = 0;
        _outDim = 0;
    }

public:
    inline void forward(Graph *cg, const vector<PNode>& x, int targetSize) {
        assert(!x.empty());
        if (x.at(0)->val.dim != _inDim) {
            std::cout << "input dim does not match for seg operation" << std::endl;
            abort();
        }

        if (_left2right) {
            left2right_forward(cg, x, targetSize);
        } else {
            right2left_forward(cg, x, targetSize);
        }
    }

protected:
    inline void left2right_forward(Graph *cg, const vector<PNode>& x, int targetSize) {
        for (int idx = 0; idx < x.size(); idx++) {
            if (idx == 0) {
                _bucket.forward(cg, 0);
                _inputgates[idx].forward(cg, &_bucket, x[idx]);
                _halfcells[idx].forward(cg, &_bucket, x[idx]);
                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);
                _cells[idx].forward(cg, &_inputfilters[idx], &_bucket);
                _halfhiddens[idx].forward(cg, &_cells[idx]);
                _outputgates[idx].forward(cg, &_bucket, x[idx]);
                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            } else if (idx == targetSize) {
                _inputgates[idx].forward(cg, &_bucket, x[idx]);
                _forgetgates[idx].forward(cg, &_bucket, x[idx]);
                _halfcells[idx].forward(cg, &_bucket, x[idx]);
                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);
                _forgetfilters[idx].forward(cg, &_cells[idx - 1], &_forgetgates[idx]);
                _cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);
                _halfhiddens[idx].forward(cg, &_cells[idx]);
                _outputgates[idx].forward(cg, &_bucket, x[idx]);
                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            } else {
                _inputgates[idx].forward(cg, &_hiddens[idx - 1], x[idx]);
                _forgetgates[idx].forward(cg, &_hiddens[idx - 1], x[idx]);
                _halfcells[idx].forward(cg, &_hiddens[idx - 1], x[idx]);
                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);
                _forgetfilters[idx].forward(cg, &_cells[idx - 1], &_forgetgates[idx]);
                _cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);
                _halfhiddens[idx].forward(cg, &_cells[idx]);
                _outputgates[idx].forward(cg, &_hiddens[idx - 1], x[idx]);
                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            }
        }
    }

    inline void right2left_forward(Graph *cg, const vector<PNode>& x, int targetSize) {
        for (int idx = x.size() - 1; idx >= 0; idx--) {
            if (idx == x.size() - 1) {
                _bucket.forward(cg, 0);
                _inputgates[idx].forward(cg, &_bucket, x[idx]);
                _forgetgates[idx].forward(cg, &_bucket, x[idx]);
                _halfcells[idx].forward(cg, &_bucket, x[idx]);
                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);
                _forgetfilters[idx].forward(cg, &_cells[0], &_forgetgates[idx]);
                _cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters.at(idx));
                _halfhiddens[idx].forward(cg, &_cells[idx]);
                _outputgates[idx].forward(cg, &_bucket, x[idx]);
                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            } else if (idx == targetSize - 1) {
                _inputgates[idx].forward(cg, &_bucket, x[idx]);
                _halfcells[idx].forward(cg, &_bucket, x[idx]);
                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);
                _cells[idx].forward(cg, &_inputfilters[idx], &_bucket);
                _halfhiddens[idx].forward(cg, &_cells[idx]);
                _outputgates[idx].forward(cg, &_bucket, x[idx]);
                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            } else {
                _inputgates[idx].forward(cg, &_hiddens[idx + 1], x[idx]);
                _forgetgates[idx].forward(cg, &_hiddens[idx + 1], x[idx]);
                _halfcells[idx].forward(cg, &_hiddens[idx + 1], x[idx]);
                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);
                _forgetfilters[idx].forward(cg, &_cells[idx + 1], &_forgetgates[idx]);
                _cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);
                _halfhiddens[idx].forward(cg, &_cells[idx]);
                _outputgates[idx].forward(cg, &_hiddens[idx + 1], x[idx]);
                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            }
        }
    }
};


#endif