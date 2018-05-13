#ifndef SRC_NNCNNLabeler_H_
#define SRC_NNCNNLabeler_H_


#include "N3LDG.h"
#include "Driver.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Classifier {
public:
    Options m_options;
    Driver m_driver;

    unordered_map<string, int> m_word_stats;

    Classifier(int memsize);
    virtual ~Classifier();

    int createAlphabet(const vector<Instance>& vecTrainInsts);
    int addTestAlpha(const vector<Instance>& vecInsts);
    void writeModelFile(const string &outputModelFile);
    void loadModelFile(const string &inputModelFile);

    void convert2Example(const Instance* pInstance, Example& exam);
    void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

    void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
    Category predict(const Feature& feature);
};

#endif /* SRC_NNCNNLabeler_H_ */
