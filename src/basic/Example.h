#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include "Targets.h"
#include "Instance.h"
#include "Category.h"

using namespace std;

class Feature
{
public:
    vector<std::string> m_title_words;

    static Feature valueOf(const Instance &ins) {
        Feature feature;
        feature.m_title_words = ins.m_title_words;
        return feature;
    }
};

class Example
{
public:
    Feature m_feature;
    Category m_category;
};

//vector<int> getClassBalancedIndexes(const std::vector<Example> &examples) {
//    std::array<std::vector<int>, 3> classSpecifiedIndexesArr;
//    for (int i = 0; i < examples.size(); ++i) {
//        const Example &example = examples.at(i);
//        classSpecifiedIndexesArr.at(example.m_category).push_back(i);
//    }

//    for (auto &v : classSpecifiedIndexesArr) {
//        std::random_shuffle(v.begin(), v.end());
//    }

//    std::array<int, 3> counters = { classSpecifiedIndexesArr.at(0).size(), classSpecifiedIndexesArr.at(1).size(), classSpecifiedIndexesArr.at(2).size() };

//    int minCounter = *std::min_element(counters.begin(), counters.end());
//    std::vector<int> indexes;

//    for (auto & v : classSpecifiedIndexesArr) {
//        for (int i = 0; i < minCounter; ++i) {
//            indexes.push_back(v.at(i));
//        }
//    }

//    std::random_shuffle(indexes.begin(), indexes.end());
//    assert(indexes.size() == 3 * minCounter);
//    return indexes;
//}

#endif /*_EXAMPLE_H_*/
