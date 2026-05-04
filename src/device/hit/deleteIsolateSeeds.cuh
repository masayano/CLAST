#ifndef C_DEVICE_HIT_LIST_DELETE_ISOLATE_SEEDS_CUH_
#define C_DEVICE_HIT_LIST_DELETE_ISOLATE_SEEDS_CUH_

#include <thrust/device_vector.h>

void deletingIsolateSeeds(
		const int allowableWidth,
		const int allowableGap,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray);

#endif
