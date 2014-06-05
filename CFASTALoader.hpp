#ifndef C_FASTA_LOADER_HPP_
#define C_FASTA_LOADER_HPP_

#include "CHostFASTA.hpp"

#include <string>
#include <vector>

class CFASTALoader {
	class CFASTAFileLoader {
		long readPos;
		long sumReadSize;
	public:
		CFASTAFileLoader(void);
		bool read(
				const std::string& mode,
				const int availableRAMSize,
				const std::string& path,
				std::vector<CHostFASTA>& FASTA);
	};
	const std::vector<std::string> fileArray;
	const int availableRAMSize;
	const std::string mode;
	CFASTAFileLoader loader;
	int fileIndex;
public:
	CFASTALoader(
			const std::vector<std::string>& fArray,
			const int avlRAMSize,
			const std::string& d);
	void loadFASTA(std::vector<CHostFASTA>& FASTA);
	int getFileIndex(void) const;
};

#endif
