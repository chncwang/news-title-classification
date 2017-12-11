#include "NNCNNLabeler.h"
#include "Stance.h"

#include <chrono> 
#include <unordered_set>
#include "Argument_helper.h"
#include "Reader.h"
#include "DomainLoss.h"

Classifier::Classifier(int memsize) : m_driver(memsize) {
    srand(0);
}

Classifier::~Classifier() {}

int Classifier::createAlphabet(const vector<Instance> &vecInsts) {
    if (vecInsts.size() == 0) {
        std::cout << "training set empty" << std::endl;
        return -1;
    }
    std::cout << "Creating Alphabet..." << endl;

    int numInstance;

    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance *pInstance = &vecInsts[numInstance];

        vector<const string *> words;
        const std::vector<std::string> &target_words = getStanceTargetWords(pInstance->m_target);
        for (const string &w : target_words) {
            words.push_back(&w);
        }

        for (const string &w : pInstance->m_tweet_words) {
            words.push_back(&w);
        }

        for (const string *w : words) {
            string normalizedWord = normalize_to_lowerwithdigit(*w);

            if (m_word_stats.find(normalizedWord) == m_word_stats.end()) {
                m_word_stats.insert(std::pair<std::string, int>(normalizedWord, 1));
            } else {
                m_word_stats.at(normalizedWord) += 1;
            }
        }

        if ((numInstance + 1) % m_options.verboseIter == 0) {
            cout << numInstance + 1 << " ";
            if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
                cout << std::endl;
            cout.flush();
        }

        if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
            break;
    }
    std::cout << numInstance << " " << endl;

    return 0;
}

int Classifier::addTestAlpha(const vector<Instance> &vecInsts) {
    std::cout << "Adding word Alphabet..." << endl;
    int numInstance;
    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance *pInstance = &vecInsts[numInstance];

        vector<const string *> words;
        const auto & target_words = getStanceTargetWords(pInstance->m_target);
        for (const string &w : target_words) {
            words.push_back(&w);
        }

        for (const string &w : pInstance->m_tweet_words) {
            words.push_back(&w);
        }

        for (const string *w : words) {
            string normalizedWord = normalize_to_lowerwithdigit(*w);

            if (m_word_stats.find(normalizedWord) == m_word_stats.end()) {
                m_word_stats.insert(std::pair<std::string, int>(normalizedWord, 0));
            } else {
                m_word_stats.at(normalizedWord) += 1;
            }
        }

        if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
            break;
    }
    cout << numInstance << " " << endl;

    return 0;
}


void Classifier::convert2Example(const Instance *pInstance, Example &exam) {
    exam.m_stance = pInstance->m_stance;
    Feature feature = Feature::valueOf(*pInstance);
    exam.m_feature = feature;
}

void Classifier::initialExamples(const vector<Instance> &vecInsts,
    vector<Example> &vecExams) {
    int numInstance;
    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance *pInstance = &vecInsts[numInstance];
        Example curExam;
        convert2Example(pInstance, curExam);
        vecExams.push_back(curExam);
    }
}

void Classifier::train(const string &trainFile, const string &devFile,
    const string &testFile, const string &modelFile,
    const string &optionFile) {
    if (optionFile != "")
        m_options.load(optionFile);
    m_options.showOptions();

    vector<Instance> rawtrainInsts = readInstancesFromFile(trainFile);
    vector<Instance> trainInsts;
    for (Instance &ins : rawtrainInsts) {
        trainInsts.push_back(ins);
    }

    std::cout << "train instances:" << std::endl;
    printStanceCount(trainInsts);

    vector<Instance> devInsts = readInstancesFromFile(devFile);
    std::cout << "dev instances:" << std::endl;
    printStanceCount(devInsts);
    vector<Instance> testInsts = readInstancesFromFile(testFile);
    std::cout << "test instances:" << std::endl;
    printStanceCount(testInsts);

    createAlphabet(trainInsts);
    if (!m_options.wordEmbFineTune) {
        addTestAlpha(devInsts);
        addTestAlpha(testInsts);
    }

    static vector<Instance> decodeInstResults;
    bool bCurIterBetter = false;

    vector<Example> trainExamples, devExamples, testExamples;

    initialExamples(trainInsts, trainExamples);
    initialExamples(devInsts, devExamples);
    initialExamples(testInsts, testExamples);

    m_word_stats[unknownkey] = m_options.wordCutOff + 1;
    m_driver._modelparams.wordAlpha.initial(m_word_stats, m_options.wordCutOff, std::unordered_set<std::string>());

    if (m_options.wordFile != "") {
        m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha,
            m_options.wordFile, m_options.wordEmbFineTune);
    } else {
        m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha,
            m_options.wordEmbSize, m_options.wordEmbFineTune);
    }

    m_driver._hyperparams.setRequared(m_options);
    m_driver.initial();

    dtype bestDIS = 0;

    srand(0);

    static vector<Example> subExamples;
    int devNum = devExamples.size(), testNum = testExamples.size();
    int non_exceeds_time = 0;
    for (int iter = 0; iter < m_options.maxIter; ++iter) {
        std::cout << "##### Iteration " << iter << std::endl;
        std::vector<int> indexes;
        if (true) {
            indexes = getClassBalancedIndexes(trainExamples);
        } else {
            for (int i = 0; i < trainExamples.size(); ++i) {
                indexes.push_back(i);
            }
            std::random_shuffle(indexes.begin(), indexes.end());
        }
        int batchBlock = indexes.size() / m_options.batchSize;
        if (indexes.size() % m_options.batchSize != 0)
            batchBlock++;
        Metric favorMetric, againstMetric, neuralMetric, overallMetric;
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
            subExamples.clear();
            int start_pos = updateIter * m_options.batchSize;
            int end_pos = (updateIter + 1) * m_options.batchSize;
            if (end_pos > indexes.size())
                end_pos = indexes.size();

            for (int idy = start_pos; idy < end_pos; idy++) {
                subExamples.push_back(trainExamples[indexes[idy]]);
            }

            int curUpdateIter = iter * batchBlock + updateIter;
            dtype cost = m_driver.train(subExamples, curUpdateIter);

            favorMetric.overall_label_count += m_driver._favor_metric.overall_label_count;
            favorMetric.correct_label_count += m_driver._favor_metric.correct_label_count;
            favorMetric.predicated_label_count += m_driver._favor_metric.predicated_label_count;
            againstMetric.overall_label_count += m_driver._against_metric.overall_label_count;
            againstMetric.correct_label_count += m_driver._against_metric.correct_label_count;
            againstMetric.predicated_label_count += m_driver._against_metric.predicated_label_count;
            neuralMetric.overall_label_count += m_driver._neural_metric.overall_label_count;
            neuralMetric.correct_label_count += m_driver._neural_metric.correct_label_count;
            neuralMetric.predicated_label_count += m_driver._neural_metric.predicated_label_count;
            m_driver.updateModel();

            if (updateIter % 10 == 1) {
                std::cout << "current: " << updateIter + 1 << ", total block: "
                    << batchBlock << std::endl;
                std::cout << "favor:" << std::endl;
                favorMetric.print();
                std::cout << "against:" << std::endl;
                againstMetric.print();
                std::cout << "neural:" << std::endl;
                neuralMetric.print();
            }
        }

        auto time_end = std::chrono::high_resolution_clock::now();
        std::cout << "Train finished. Total time taken is: "
            << std::chrono::duration<double>(time_end - time_start).count()
            << "s" << std::endl;
        float accuracy = static_cast<float>(favorMetric.correct_label_count + againstMetric.correct_label_count + neuralMetric.correct_label_count) /
            (favorMetric.overall_label_count + againstMetric.overall_label_count + neuralMetric.overall_label_count);
        std::cout << "train set acc:" << accuracy << std::endl;
        if (accuracy >= 0.95) {
            std::cout << "train set is good enough, stop" << std::endl;
            exit(0);
        }

        float devAvg = 0.0;
        if (devNum > 0) {
            Metric favor, against;
            auto time_start = std::chrono::high_resolution_clock::now();
            bCurIterBetter = false;
            if (!m_options.outBest.empty())
                decodeInstResults.clear();
            for (int idx = 0; idx < devExamples.size(); idx++) {
                int excluded_class = -1;
                if (m_options.postProcess) {
                    excluded_class = isTargetWordInTweet(devExamples.at(idx).m_feature.m_target, devExamples.at(idx).m_feature.m_tweet_words) ? Stance::NONE : -1;
                }
                Stance result = predict(devExamples[idx].m_feature, excluded_class);

                devInsts[idx].evaluate(result, favor, against);

                if (!m_options.outBest.empty()) {
                    decodeInstResults.push_back(devInsts[idx]);
                }
            }

            auto time_end = std::chrono::high_resolution_clock::now();
            std::cout << "Dev finished. Total time taken is: "
                << std::chrono::duration<double>(time_end - time_start).count()
                << "s" << std::endl;
            std::cout << "dev:" << std::endl;
            std::cout << "favor:" << std::endl;
            favor.print();
            std::cout << "against:" << std::endl;
            against.print();

            if (!m_options.outBest.empty() > bestDIS) {
                /*m_pipe.outputAllInstances(devFile + m_options.outBest,
                  decodeInstResults);*/
                bCurIterBetter = true;
            }

            float testAvg = 0;
            if (testNum > 0) {
                auto time_start = std::chrono::high_resolution_clock::now();
                if (!m_options.outBest.empty())
                    decodeInstResults.clear();
                Metric favor, against;
                for (int idx = 0; idx < testExamples.size(); idx++) {
                    int excluded_class = -1;
                    if (m_options.postProcess) {
                        excluded_class = isTargetWordInTweet(testExamples.at(idx).m_feature.m_target, testExamples.at(idx).m_feature.m_tweet_words) ? Stance::NONE : -1;
                    }
                    Stance stance = predict(testExamples[idx].m_feature, excluded_class);

                    testInsts[idx].evaluate(stance, favor, against);

                    if (bCurIterBetter && !m_options.outBest.empty()) {
                        decodeInstResults.push_back(testInsts[idx]);
                    }
                }

                auto time_end = std::chrono::high_resolution_clock::now();
                std::cout << "Test finished. Total time taken is: "
                    << std::chrono::duration<double>(
                        time_end - time_start).count() << "s" << std::endl;
                std::cout << "test:" << std::endl;
                std::cout << "favor:" << std::endl;
                favor.print();
                std::cout << "against:" << std::endl;
                against.print();
                testAvg = (favor.getFMeasure() + against.getFMeasure()) * 0.5;
                std::cout << "avg f:" << testAvg << std::endl;

                /*if (!m_options.outBest.empty() && bCurIterBetter) {
                  m_pipe.outputAllInstances(testFile + m_options.outBest,
                  decodeInstResults);
                  }*/
            }

            double avgFMeasure = (favor.getFMeasure() + against.getFMeasure()) * 0.5;
            if (m_options.saveIntermediate && avgFMeasure > bestDIS) {
                std::cout << "Exceeds best previous performance of " << bestDIS
                    << " now is " << avgFMeasure << ". Saving model file.." << std::endl;
                std::cout << "laozhongyi_" << std::min<float>(avgFMeasure, testAvg) << std::endl;
                non_exceeds_time = 0;
                bestDIS = avgFMeasure;
                writeModelFile(modelFile);
            }
        }
        // Clear gradients
    }
}

Stance Classifier::predict(const Feature &feature, int excluded_class) {
    //assert(features.size() == words.size());
    Stance stance;
    m_driver.predict(feature, stance, excluded_class);
    return stance;
}

void Classifier::test(const string &testFile, const string &outputFile,
    const string &modelFile) {
    loadModelFile(modelFile);
    m_driver.TestInitial();
    vector<Instance> testInsts = readInstancesFromFile(testFile);

    vector<Example> testExamples;
    initialExamples(testInsts, testExamples);

    int testNum = testExamples.size();
    vector<Instance> testInstResults;
    Metric favor, against;
    for (int idx = 0; idx < testExamples.size(); idx++) {
        Stance stance = predict(testExamples[idx].m_feature, -1);
        testInsts[idx].evaluate(stance, favor, against);
        Instance curResultInst;
        //curResultInst.assignLabel(result_label);
        testInstResults.push_back(testInsts[idx]);
    }
    std::cout << "test:" << std::endl;
    std::cout << "favor:" << std::endl;
    favor.print();
    std::cout << "against:" << std::endl;
    against.print();

    //m_pipe.outputAllInstances(outputFile, testInstResults);
}


void Classifier::loadModelFile(const string &inputModelFile) {
    ifstream is(inputModelFile);
    if (is.is_open()) {
        m_driver._hyperparams.loadModel(is);
        m_driver._modelparams.loadModel(is);
        is.close();
    } else
        std::cout << "load model error" << endl;
}

void Classifier::writeModelFile(const string &outputModelFile) {
    ofstream os(outputModelFile);
    if (os.is_open()) {
        m_driver._hyperparams.saveModel(os);
        m_driver._modelparams.saveModel(os);
        os.close();
        std::cout << "write model ok. " << endl;
    } else
        std::cout << "open output file error" << endl;
}

#include "Targets.h"

//int main(int argc, char *argv[]) {
//	vector<Instance> instances = readInstancesFromFile("C:/N3LDGStanceDetector/data/SemEval2016-Task6-subtaskB-testdata-gold.txt");
//
//	for (Instance &ins : instances) {
//		for (const string &w : ins.m_target_words) {
//			std::cout << w << " ";
//		}
//		std::cout << std::endl;
//		for (string &w : ins.m_tweet_words) {
//			std::cout << w << "|";
//		}
//		std::cout << std::endl;
//	}
//
//	while (true);
//	return 0;
//}


int main(int argc, char *argv[]) {
    std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
    std::string outputFile = "";
    bool bTrain = false;
    int memsize = 0;
    dsr::Argument_helper ah;

    ah.new_flag("l", "learn", "train or test", bTrain);
    ah.new_named_string("train", "trainCorpus", "named_string",
        "training corpus to train a model, must when training", trainFile);
    ah.new_named_string("dev", "devCorpus", "named_string",
        "development corpus to train a model, optional when training", devFile);
    ah.new_named_string("test", "testCorpus", "named_string",
        "testing corpus to train a model or input file to test a model, optional when training and must when testing",
        testFile);
    ah.new_named_string("model", "modelFile", "named_string",
        "model file, must when training and testing", modelFile);
    ah.new_named_string("option", "optionFile", "named_string",
        "option file to train a model, optional when training", optionFile);
    ah.new_named_string("output", "outputFile", "named_string",
        "output file to test, must when testing", outputFile);
    ah.new_named_int("memsize", "memorySize", "named_int",
        "This argument decides the size of static memory allocation", memsize);

    ah.process(argc, argv);

    if (memsize < 0)
        memsize = 0;
    Classifier the_classifier(memsize);
    if (bTrain) {
        the_classifier.train(trainFile, devFile, testFile, modelFile, optionFile);
    } else {
        the_classifier.test(testFile, outputFile, modelFile);
    }
    //getchar();
    //test(argv);
    //ah.write_values(std::cout);
}
