#ifndef C_HOST_SCHEDULAR_CUH_
#define C_HOST_SCHEDULAR_CUH_

#include "CHostResultHolder.cuh"
#include "CHostSetting.cuh"
#include "CHostSeqList_query.cuh"
#include "CHostSeqList_target.cuh"

class CHostSchedular {
	const CHostSetting&    setting;
	const CHostSeqList_target& targetList;
	const CHostSeqList_query&  queryList;
public:
	CHostSchedular(
			const CHostSetting& s,
			const CHostSeqList_target& t,
			const CHostSeqList_query& q);
	void search(CHostResultHolder& holder);
};

#endif
