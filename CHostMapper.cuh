#ifndef C_MAPPER_CUH_
#define C_MAPPER_CUH_

#include "CHostFASTA.hpp"
#include "CHostSeqList_query.cuh"
#include "CHostSeqList_target.cuh"
#include "CHostResultHolder.cuh"
#include "CHostSetting.cuh"

#include <string>
#include <vector>

class CHostMapper {
	const CHostSetting& setting;
	CHostSeqList_target targetList;
	CHostSeqList_query  queryList;
public:
	CHostMapper(const CHostSetting& s);
	void addTarget(const std::vector<CHostFASTA>& t);
	void addQuery (const std::vector<CHostFASTA>& q);
	void getResult(CHostResultHolder& holder) const;
};

#endif
