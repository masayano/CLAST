#ifndef C_HOST_FASTA_HPP_
#define C_HOST_FASTA_HPP_

#include <string>

class CHostFASTA {
	std::string label;
	std::string sequence;
	int startIdx;
public:
	CHostFASTA(
			const std::string& l,
			const std::string& s,
			const int sIdx);
	const std::string& getLabel   (void) const;
	const std::string& getSequence(void) const;
	int getStartIdx(void) const;
};

#endif /* C_HOST_FASTA_HPP_ */
