#ifndef C_DEVICE_HIT_LIST_DELETE_DUPLICATE_SEEDS_CUH_
#define C_DEVICE_HIT_LIST_DELETE_DUPLICATE_SEEDS_CUH_

#include "host/CHostSetting.cuh"

#include <thrust/device_vector.h>

void deleteDuplicateSeeds(
		const CHostSetting& s,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray);

#endif
