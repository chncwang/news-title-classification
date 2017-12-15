#ifndef _JST_READER_
#define _JST_READER_

#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <regex>

#include "Instance.h"
#include "Targets.h"

std::vector<std::string> readLines(const std::string &fullFileName) {
    std::vector<string> lines;
    std::ifstream input(fullFileName);
    for (std::string line; getline(input, line);) {
        lines.push_back(line);
    }
    return lines;
}

void readLineToInstance(const std::string &line, Instance *instance) {
    int index = line.find_first_of(",");
    assert(index != std::string::npos);
    std::string category_str = line.substr(0, index);
    Category category = ToCategory(category_str);
    instance->m_category = category;
    std::string title = line.substr(index);
    std::vector<std::string> words;
    split_bychars(title, words);
    instance->m_title_words = std::move(words);
}

std::vector<Instance> readInstancesFromFile(const std::string &fullFileName) {
    std::vector<std::string> lines = readLines(fullFileName);
    std::vector<Instance> instances;
    for (const std::string &line : lines) {
        Instance instance;
        readLineToInstance(line, &instance);
        instances.push_back(instance);
    }

    return instances;
}

#endif
