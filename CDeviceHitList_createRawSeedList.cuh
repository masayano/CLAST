#ifndef C_DEVICE_HIT_LIST_CREATE_RAW_SEED_LIST_CUH_
#define C_DEVICE_HIT_LIST_CREATE_RAW_SEED_LIST_CUH_

#include "CHostSetting.cuh"
#include "CDeviceHashTable.cuh"
#include "CDeviceSeqList_query.cuh"

#include <thrust/device_vector.h>

void createRawSeedList(
		const CHostSetting& s,
		const CDeviceHashTable& h,
		const CDeviceSeqList_query& q,
		const int t_begin,
		const int q_begin,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray);

#endif
