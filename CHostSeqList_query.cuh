#ifndef C_HOST_SEQ_LIST_QUERY_CUH_
#define C_HOST_SEQ_LIST_QUERY_CUH_

#include "CHostSeqList.cuh"

class CHostSeqList_query : public CHostSeqList {
public:
	void add(
		const CHostSetting& setting,
		const std::vector<CHostFASTA>& seq);
};

#endif
