#ifndef _JST_READER_
#define _JST_READER_

#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <regex>
using namespace std;

#include "Instance.h"
#include "Targets.h"

vector<string> readLines(const string &fullFileName) {
    vector<string> lines;
    std::ifstream input(fullFileName);
    for (std::string line; getline(input, line);) {
        lines.push_back(line);
    }
    return lines;
}

void readLineToInstance(const string &line, Instance *instance) {
    //cout << "Reader readLineToInstance line:" << line << endl;
    int tailIndex = -1;
    int i = 0;
    auto targetWordVectors = getStanceTargetWordVectors();
    for (const string &target : getStanceTargets()) {
        string::size_type index = line.find(target);
        if (index <= 8) {
            string firstWord = targetWordVectors.at(i).at(0);
            if (firstWord == "Atheism") {
                instance->m_target = Target::ATHEISM;
            } else if (firstWord == "Climate") {
                instance->m_target = Target::CLIMATE_CHANGE;
            } else if (firstWord == "Feminist") {
                instance->m_target = Target::FEMINIST_MOVEMENT;
            } else if (firstWord == "Hillary") {
                instance->m_target = Target::HILLARY_CLINTON;
            } else if (firstWord == "Legalization") {
                instance->m_target = Target::LEGALIZATION_OF_ABORTION;
            } else if (firstWord == "Donald") {
                instance->m_target = Target::DONALD_TRUMP;
            } else {
                std::cout << firstWord << std::endl;
                abort();
            }

            tailIndex = index + target.size();
            break;
        }
        ++i;
    }

    if (tailIndex == -1) {
        cout << "target not found!" << line << endl;
        assert(false);
    }

    string::size_type index = string::npos;
    for (int i = 0; i < 3; ++i) {
        Stance stance = static_cast<Stance>(i);
        const string &stanceStr = StanceToString(stance);
        std::regex regex(stanceStr + "\r?$");
        for (auto it = std::sregex_iterator(line.begin(), line.end(), regex);
            it != std::sregex_iterator();
            ++it) {
            index = it->position();
            instance->m_stance = stance;
            break;
        }
    }
    if (index == string::npos) {
        std::cout << line << std::endl;
        abort();
    }

    assert(index != string::npos);

    string substring = line.substr(tailIndex, index - tailIndex);
    //std::cout << "Reader readLineToInstance substring:" << substring << endl;

    //std::regex regex("[\s\t]+(.+)");
    //std::smatch matcher;
    //if (!std::regex_search(substring, matcher, regex)) {
    //	//std::cout << "Reader readLineToInstance regex not found!" << endl;
    //	std::cout << "substring:" << substring << std::endl;
    //	assert(false);
    //}

    //string sentence = matcher.format("$1");
    //std::cout << "Reader readLineToInstance sentence:" << sentence << "|||" << endl;

    vector<string> rawwords;
    split_bychar(substring, rawwords);
    vector<string> words;
    for (string & rawword : rawwords) {
        if (rawword.empty()) continue;
        string word = normalize_to_lowerwithdigit(rawword);
        if (word == "rt" || word == "via" || word == "#semst") continue;
        //if (word == "thats" || word == "im" || word == "'s") continue;
        if (isPunctuation(word)) continue;

        std::string http = "http";
        if (!word.compare(0, http.size(), http)) {
            continue;
        }

        words.push_back(word);
    }

    assert(!words.empty());

    //std::cout << instance->m_stance << std::endl;
    instance->m_tweet_words = move(words);
}

vector<Instance> readInstancesFromFile(const string &fullFileName) {
    vector<string> lines = readLines(fullFileName);

    vector<Instance> instances;
    using std::move;
    for (int i = 0; i < lines.size(); ++i) {
        if (lines.at(i) == "ID Target Tweet Stance") continue;
        Instance ins;
        readLineToInstance(lines.at(i), &ins);
        instances.push_back(move(ins));
    }

    return instances;
}

#endif
