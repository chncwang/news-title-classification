#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
#include "Targets.h"
#include "Category.h"

using namespace std;

class Instance
{
public:
    void evaluate(Category predicted_category, Metric& metric) const {
        ++metric.overall_label_count;
        if (predicted_category == m_category) {
            metric.correct_label_count++;
        }
    }

    int size() const {
        return m_title_words.size();
    }

    std::string tostring();
public:
    vector<string> m_title_words;
    Category m_category;
};

std::string Instance::tostring() {
    string result = "target: ";

    for (string & w : m_title_words) {
        result += w + " ";
    }
    result += "\nstance: " + m_category;
    return result;
}

#endif /*_INSTANCE_H_*/
