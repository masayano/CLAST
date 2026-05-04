#ifndef C_DEVICE_HIT_LIST_DELETE_SEEDS_ON_SEQUENCE_BOUNDARY_CUH_
#define C_DEVICE_HIT_LIST_DELETE_SEEDS_ON_SEQUENCE_BOUNDARY_CUH_

#include "host/CHostSetting.cuh"
#include "device/CDeviceHashTable.cuh"
#include "device/seq/query.cuh"

#include <thrust/device_vector.h>

void deleteSeedsOnSequenceBoundary(
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
