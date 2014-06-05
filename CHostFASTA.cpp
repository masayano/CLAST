#include "CHostFASTA.hpp"

CHostFASTA::CHostFASTA(
		const std::string& l,
		const std::string& s,
		const int sIdx)
		: label(l), sequence(s), startIdx(sIdx) {}

const std::string& CHostFASTA::getLabel   (void) const { return label; }
const std::string& CHostFASTA::getSequence(void) const { return sequence; }
int CHostFASTA::getStartIdx(void) const { return startIdx; }
