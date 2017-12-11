#ifndef STANCE_DETECTOR_STANCE_H
#define STANCE_DETECTOR_STANCE_H

#include <string>
#include <array>

enum Stance {
	AGAINST = 0,
	FAVOR = 1,
	NONE = 2
};

const string& StanceToString(Stance stance) {
	static const std::array<string, 3> STANCE_STRS = {"AGAINST", "FAVOR", "NONE"};
	return STANCE_STRS.at(stance);
}


#endif