#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
#include "Targets.h"
#include "Stance.h"

using namespace std;

class Instance
{
public:
	void evaluate(Stance predict_stance, Metric& favorMetric, Metric &againstMetric) const
	{
		if (m_stance == Stance::FAVOR) {
			favorMetric.overall_label_count++;
		}
		else if (m_stance == Stance::AGAINST) {
			againstMetric.overall_label_count++;
		}

		if (predict_stance == Stance::FAVOR) {
			favorMetric.predicated_label_count++;
			if (m_stance == Stance::FAVOR) {
				favorMetric.correct_label_count++;
			}
		} else if (predict_stance == Stance::AGAINST) {
			againstMetric.predicated_label_count++;
			if (m_stance == Stance::AGAINST) {
				againstMetric.correct_label_count++;
			}
		}
	}

	int size() const {
		return m_tweet_words.size();
	}

	std::string tostring();
public:
	vector<string> m_tweet_words;
	Stance m_stance;
    Target m_target;
};

std::string Instance::tostring() {
	string result = "target: ";

	for (string & w : m_tweet_words) {
		result += w + " ";
	}
	result += "\nstance: " + StanceToString(m_stance);
	return result;
}

void printStanceCount(const vector<Instance> &instances) {
	int favorCount = 0;
	int againstCount = 0;
	int neutralCount = 0;
	for (const Instance &ins : instances) {
		if (ins.m_stance == Stance::FAVOR) {
			favorCount++;
		}
		else if (ins.m_stance == Stance::AGAINST) {
			againstCount++;
		}
		else if (ins.m_stance == Stance::NONE) {
			neutralCount++;
		}
		else {
			abort();
		}
	}

	std::cout << "favor: " << favorCount << " against: " << againstCount << " neutral: " << neutralCount << std::endl;
}

#endif /*_INSTANCE_H_*/
