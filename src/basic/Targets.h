#ifndef SRC_BASIC_TARGETS_H
#define SRC_BASIC_TARGETS_H

#include <vector>
#include <array>
#include <string>
#include "MyLib.h"
#include "Example.h"

enum Target {
    ATHEISM = 0,
    CLIMATE_CHANGE = 1,
    FEMINIST_MOVEMENT = 2,
    HILLARY_CLINTON = 3,
    LEGALIZATION_OF_ABORTION = 4,
    DONALD_TRUMP = 5
};


constexpr int DOMAIN_SIZE = 5;
const std::array<Target, DOMAIN_SIZE> DOMAIN_TARGETRS = {ATHEISM, CLIMATE_CHANGE, FEMINIST_MOVEMENT, HILLARY_CLINTON, LEGALIZATION_OF_ABORTION};


const std::vector<string> &getStanceTargets() {
    static std::vector<std::string> targets = { "Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion", "Donald Trump" };
    return targets;
}

std::vector<vector<string>> getStanceTargetWordVectors() {
    using std::move;
    auto &targets = getStanceTargets();
    std::vector<vector<string> > result;
    for (const std::string & str : targets) {
        vector<string> words;
        split_bychar(str, words);
        result.push_back(move(words));
    }

    return result;
}

const std::vector<std::string> &getStanceTargetWords(Target target) {
    //static std::vector<std::string> ATHEISM = { "atheism" };
    //static std::vector<std::string> CLIMATE_CHANGE = { "climate", "change", "is", "a", "real", "concern" };
    //static std::vector<std::string> FEMINIST_MOVEMENT = { "feminist", "movement" };
    //static std::vector<std::string> HILLARY_CLINTON = { "hillary", "clinton" };
    //static std::vector<std::string> LEGALIZATION_OF_ABORTION = { "legalization", "of", "abortion" };
    //static std::vector<std::string> DONALD_TRUMP = { "donald", "trump" };
    static std::vector<std::string> ATHEISM = { "#atheism" };
    static std::vector<std::string> CLIMATE_CHANGE = { "#climatechange"};
    static std::vector<std::string> FEMINIST_MOVEMENT = { "#feminism" };
    static std::vector<std::string> HILLARY_CLINTON = { "#hillaryclinton" };
    static std::vector<std::string> LEGALIZATION_OF_ABORTION = { "#prochoice" };
    static std::vector<std::string> DONALD_TRUMP = { "#donaldtrump" };
    static std::array<std::vector<std::string>, 6> ARR = { ATHEISM, CLIMATE_CHANGE, FEMINIST_MOVEMENT, HILLARY_CLINTON, LEGALIZATION_OF_ABORTION, DONALD_TRUMP };
    return ARR.at(target);
}

bool isTargetWordInTweet(Target target, const std::vector<std::string> &tweet) {
    std::vector<std::string> keywords;
    if (target == Target::HILLARY_CLINTON) {
        keywords = { "hillary", "clinton" };
    } else if (target == Target::DONALD_TRUMP) {
        keywords = { "donald", "trump" };
    } else if (target == Target::ATHEISM) {
        keywords = { "atheism", "atheist" };
    } else if (target == Target::CLIMATE_CHANGE) {
        keywords = { "climate" };
    } else if (target == Target::FEMINIST_MOVEMENT) {
        keywords = { "feminism", "feminist" };
    } else if (target == Target::LEGALIZATION_OF_ABORTION) {
        keywords = { "abortion", "aborting" };
    } else {
        abort();
    }
    for (const std::string &keyword : keywords) {
        for (const std::string &tweetword : tweet) {
            if (tweetword.find(keyword) != std::string::npos) {
                return true;
            }
        }
    }

    return false;
}

#endif // !TARGETS_H
