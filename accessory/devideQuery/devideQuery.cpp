#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <iomanip>

#include <boost/lexical_cast.hpp>

const int selectMinIdx(std::vector<long>& lengthArray) {
	int idx = 0;
	for(int i = 0; i < lengthArray.size(); ++i) {
		if(lengthArray[idx] > lengthArray[i]) {
			idx = i;
		}
	}
	return idx;
}

void writeFASTA(
		const std::string& label,
		const std::string& sequence,
		const std::string& devideFASTA,
		std::vector<long>& lengthArray) {
	const int idx = selectMinIdx(lengthArray);
	std::stringstream fileName;
	fileName << devideFASTA << "." << idx;
	std::ofstream ofs(fileName.str().c_str(), std::ios::app);
	ofs << label << std::endl;
	ofs << sequence << std::endl << std::endl;
	lengthArray[idx] += sequence.size();
}

void eraseData(std::string& label, std::string& sequence) {
        label   .erase();
        sequence.erase();
}

void removeSpace(std::string& str) {
        std::string::size_type pos = 0;
        while(pos = str.find(" ", pos), pos != std::string::npos) {
                str.replace(pos, 1, "");
        }
}

void loadFASTA(
		const std::string& devideFASTA,
		std::vector<long>& lengthArray) {
        std::ifstream ifs(devideFASTA.c_str());
        std::string buf;

        std::string label;
        std::string sequence;
        while(ifs && std::getline(ifs, buf)) {
                if(buf.find(">") != std::string::npos) {
                        if(!label.empty()) {
                                writeFASTA(label, sequence, devideFASTA, lengthArray);
                                eraseData(label, sequence);
                        }
                        label = buf;
                } else if(buf.empty()) {
                        if(!label.empty()) {
                                writeFASTA(label, sequence, devideFASTA, lengthArray);
                                eraseData(label, sequence);
                        }
                } else {
                        if(!label.empty()) {
                                removeSpace(buf);
                                sequence += buf;
                        }
                }
        }
        if(ifs.eof()) {
                if(!label.empty()) {
                        writeFASTA(label, sequence, devideFASTA, lengthArray);
                        eraseData(label, sequence);
                }
        }
}

int main(int argc, char** argv) {
	if(argc != 3) {
		std::cout << "arg1: devide number." << std::endl;
		std::cout << "arg2: devide FASTA file." << std::endl;
		abort();
	}
	std::vector<long> lengthArray(boost::lexical_cast<int>(argv[1]), 0);
	const std::string devideFASTA = argv[2];

	loadFASTA(devideFASTA, lengthArray);

	return 0;
}
