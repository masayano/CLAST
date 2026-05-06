#include "util/fastaStringUtil.hpp"

void removeSpace(std::string& str) {
	std::string::size_type pos = 0;
	while (pos = str.find(" ", pos), pos != std::string::npos) {
		str.replace(pos, 1, "");
	}
}

void replaceSmallToBig(std::string& str) {
	const char* end = (&(str[0])) + str.size();
	for (char* temp = &(str[0]); temp < end; ++temp) {
		if (*temp > 'Z') {
			*temp -= 32;
		}
		if (*temp < 'A' || *temp > 'Z') {
			*temp = 'N';
		}
	}
}
