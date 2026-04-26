#ifndef C_MAPPER_CUH_
#define C_MAPPER_CUH_

#include "host/CHostFASTA.hpp"
#include "host/seq/query.cuh"
#include "host/seq/target.cuh"
#include "host/CHostResultHolder.cuh"
#include "host/CHostSetting.cuh"

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
