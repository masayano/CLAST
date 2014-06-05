#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <boost/lexical_cast.hpp>

void writeFASTA(
		const std::string& outPath,
		const std::string& label,
		const int startIdx,
		const std::string& outputSequence) {
	std::ofstream ofs(outPath.c_str(), std::ios::app);

	ofs << label << std::endl;
	ofs << "@" << startIdx << std::endl;

	int i;
	for(i = 0; i <= (static_cast<int>(outputSequence.size()) - 60); i += 60) {
		ofs << outputSequence.substr(i, 60) << std::endl;
	}
	ofs << outputSequence.substr(i) << std::endl << std::endl;
}

void splitFASTA(
		const std::string& label,
		const std::string& sequence,
		const std::string& outPath,
		const int maxLength,
		const int overlapLength) {
	int i;
	std::string outputSequence;
	for(i = 0; i <= (static_cast<int>(sequence.size()) - maxLength); i += (maxLength - overlapLength)) {
		outputSequence = sequence.substr(i, maxLength);
		writeFASTA(outPath, label, i, outputSequence);
	}
	outputSequence = sequence.substr(i);
	writeFASTA(outPath, label, i, outputSequence);
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

void readFASTA(
		const std::string& inPath,
		const std::string& outPath,
		const int maxLength,
		const int overlapLength) {
	std::ifstream ifs(inPath.c_str());
	std::string buf;

	std::string label;
	std::string sequence;
	while(ifs && std::getline(ifs, buf)) {
		if(buf.find(">") != std::string::npos) {
			if(!label.empty()) {
				splitFASTA(label, sequence, outPath, maxLength, overlapLength);
				eraseData(label, sequence);
			}
			label = buf;
		} else if(buf.empty()) {
			if(!label.empty()) {
				splitFASTA(label, sequence, outPath, maxLength, overlapLength);
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
			splitFASTA(label, sequence, outPath, maxLength, overlapLength);
			eraseData(label, sequence);
		}
	}
}

int main(int argc, char** argv) {
	if(argc == 5) {
		const std::string inPath  = argv[1];
		const std::string outPath = argv[2];
		const int maxLength     = boost::lexical_cast<int>(argv[3]);
		const int overlapLength = boost::lexical_cast<int>(argv[4]);
		if(maxLength <= overlapLength) {
			std::cout << " arg3(after splitted length) must be bigger than arg4(overlap length)." << std::endl;
			abort();
		}
		readFASTA(inPath, outPath, maxLength, overlapLength);
	} else {
		std::cout
			<< "This program splits large reference genome into many small parts." << std::endl
			<< std::endl
			<< " <------------------------ large reference genome ------------------------>" << std::endl
			<< " <-------- small part -------->" << std::endl
			<< "                          <-------- small part -------->" << std::endl
			<< "                                                   <----- small part ----->" << std::endl
			<< std::endl
			<< "usage:" << std::endl
			<< " arg1 -> input reference fasta file path." << std::endl
			<< " arg2 -> output file path." << std::endl
			<< " arg3 -> after splitted length." << std::endl
			<< " arg4 -> overlap length between parts." << std::endl;
	}
	return 0;
}
