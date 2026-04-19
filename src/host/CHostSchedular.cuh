#ifndef C_HOST_SCHEDULAR_CUH_
#define C_HOST_SCHEDULAR_CUH_

#include "host/CHostResultHolder.cuh"
#include "host/CHostSetting.cuh"
#include "host/CHostSeqList_query.cuh"
#include "host/CHostSeqList_target.cuh"

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
