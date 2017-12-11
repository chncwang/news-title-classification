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
}

vector<Instance> readInstancesFromFile(const string &fullFileName) {
}

#endif
