#include "util/fastaDataSizeUtil.hpp"

#include <fstream>
#include <string>

long calcTotalDbSizeFromFastaPaths(const std::vector<std::string>& filePaths) {
	long total = 0;
	for (const auto& path : filePaths) {
		std::ifstream ifs(path.c_str(), std::ios::binary);
		std::string buf;
		while (ifs && std::getline(ifs, buf)) {
			if (buf.find('>') == std::string::npos) {
				total += static_cast<long>(buf.size());
			}
		}
	}
	return total;
}
