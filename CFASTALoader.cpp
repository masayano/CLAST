#include "CFASTALoader.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>

#include <boost/lexical_cast.hpp>

/******************************************* class function *******************************************/

CFASTALoader::CFASTALoader(
		const std::vector<std::string>& fArray,
		const int avlRAMSize,
		const std::string& m)
		: fileArray(fArray),
		  availableRAMSize(avlRAMSize),
		  mode(m),
		  fileIndex(0){
	if((mode != "target") && (mode != "query")) {
		std::cout
				<< " at CFASTALoader::loadFASTA() : bad argument (const std::string& mode)." << std::endl
				<< " ...The argument was : " << mode << "." << std::endl;
		abort();
	}
}

void CFASTALoader::loadFASTA(std::vector<CHostFASTA>& FASTA) {
	for(; fileIndex < static_cast<int>(fileArray.size()); ++fileIndex) {
		if(!loader.read(mode, availableRAMSize, fileArray[fileIndex], FASTA)) {
			break;
		}
	}
	if(fileIndex == static_cast<int>(fileArray.size())) {
		fileIndex = -1;
	}
}

int CFASTALoader::getFileIndex() const { return fileIndex; }

/************************** non CFASTALoader::CFASTAFileLoader class function ***************************/
#include "common.hpp"

bool judgeContinueReading(
		const std::string& mode,
		const long pos,
		const int availableRAMSize,
		std::string& label,
		std::string& sequence,
		int& startIdx,
		std::vector<CHostFASTA>& FASTA,
		long& sumReadSize,
		long& readPos) {
	if(!label.empty()) {
		const int sequenceLength = sequence.size();
		if(sequenceLength > availableRAMSize) {
			if(mode == "target") {
				std::cout
						<< " warning : Reference sequence " << label << " is longer than \"CPU-RAM usage for target\"." << std::endl
						<< " No query will be mapped to this sequence." << std::endl
						<< " You can avoid it if you preprocess the reference fasta file." << std::endl
						<< " Prease read READ ME." << std::endl;
			} else /* mode == "query" */ {
				std::cout
						<< " warning : Reference sequence " << label << " is longer than \"CPU-RAM usage for query\"." << std::endl
						<< " This query will not be mapped to any reference sequence." << std::endl;
			}
			label   .erase();
			sequence.erase();
			startIdx = 0;
			readPos = pos;
		} else if(sumReadSize + sequenceLength < availableRAMSize) {
			/* add fasta */ {
				CHostFASTA fasta(label, sequence, startIdx);
				FASTA.push_back(fasta);
				sumReadSize += sequenceLength;
			}
			label   .erase();
			sequence.erase();
			startIdx = 0;
			readPos = pos;
		} else {
			sumReadSize = 0;
			return false;
		}
	}
	return true;
}

void removeSpace(std::string& str) {
	std::string::size_type pos = 0;
	while(pos = str.find(" ", pos), pos != std::string::npos) {
		str.replace(pos, 1, "");
	}
}

void replaceSmallToBig(std::string& str) {
	std::string::size_type pos = 0;
	while(pos = str.find("a", pos), pos != std::string::npos) {
		str.replace(pos, 1, "A");
	}
	pos = 0;
	while(pos = str.find("c", pos), pos != std::string::npos) {
		str.replace(pos, 1, "C");
	}
	pos = 0;
	while(pos = str.find("g", pos), pos != std::string::npos) {
		str.replace(pos, 1, "G");
	}
	pos = 0;
	while(pos = str.find("t", pos), pos != std::string::npos) {
		str.replace(pos, 1, "T");
	}
}

/******************************* CFASTALoader::CFASTAFileLoader function ********************************/

CFASTALoader::CFASTAFileLoader::CFASTAFileLoader(void) : readPos(0), sumReadSize(0) {}

bool CFASTALoader::CFASTAFileLoader::read(
		const std::string& mode,
		const int availableRAMSize,
		const std::string& path,
		std::vector<CHostFASTA>& FASTA) {
	std::ifstream ifs(path.c_str(), std::ios::binary);
	if(!ifs) {
		std::cout << "file \"" << path << "\" open failed." << std::endl;
		abort();
	}
	std::string buf;

	std::string label;
	std::string sequence;
	int startIdx = 0;

	ifs.seekg(readPos);
	long tempPos = readPos;
	while(ifs && std::getline(ifs, buf)) {
		if(buf.find(">") != std::string::npos) {
			if(!judgeContinueReading(
					mode, tempPos, availableRAMSize,
					label, sequence, startIdx,
					FASTA, sumReadSize, readPos)) { return false; }
			label = buf.substr(1);
		} else if(buf.find("@") != std::string::npos) {
			try {
				startIdx = boost::lexical_cast<int>(buf.substr(1)); // for clast-modified form fasta.
			} catch(boost::bad_lexical_cast) {
				std::cout
					<< " at CFASTALoader::CFASTAFilteLoader::read() : bad file type." << std::endl
					<< " Something was strange at the line which contains '@'." << std::endl;
				abort();
			}
		} else if(buf.empty()) {
			if(!judgeContinueReading(
					mode, tempPos, availableRAMSize,
					label, sequence, startIdx,
					FASTA, sumReadSize, readPos)) { return false; }
		} else {
			if(!label.empty()) {
				removeSpace(buf);
				replaceSmallToBig(buf);
				sequence += buf;
			}
		}
		tempPos = ifs.tellg();
 	}
	if(ifs.eof()) {
		if(!judgeContinueReading(
				mode, tempPos, availableRAMSize,
				label, sequence, startIdx,
				FASTA, sumReadSize, readPos)) {
			return false;
		}
	}
	readPos = 0;
	return true;
}
