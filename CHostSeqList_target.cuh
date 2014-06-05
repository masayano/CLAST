#ifndef C_HOST_SEQ_LIST_TARGET_CUH_
#define C_HOST_SEQ_LIST_TARGET_CUH_

#include "CHostSeqList.cuh"

class CHostSeqList_target : public CHostSeqList {
	std::vector<int> startIdxArray;
public:
	void add(
			const CHostSetting& setting,
			const std::vector<CHostFASTA>& seq);
	int getStartIdx(const int seqID) const;
};

#endif
