#ifndef C_DEVICE_HIT_LIST_SEED_HOST_API_CUH_
#define C_DEVICE_HIT_LIST_SEED_HOST_API_CUH_

#include "device/hit/seedThrustVectorOps.cuh"

#include <thrust/host_vector.h>

inline void deleteDuplicateSeeds(
		const int allowableWidth,
		const int allowableGap,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
	clast::hit::deleteDuplicateSeeds(
			allowableWidth,
			allowableGap,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
}

inline void deleteSeedHasNotNearPair(
		const int allowableWidth,
		const int allowableGap,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
	clast::hit::deleteSeedHasNotNearPair(
			allowableWidth,
			allowableGap,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
}

inline void onQueryBoundary(
		const int kMerLength,
		const int q_begin,
		const thrust::host_vector<int>& qLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
	clast::hit::onQueryBoundary(
			kMerLength,
			q_begin,
			qLengthArray,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
}

inline void onTargetBoundary(
		const int kMerLength,
		const int t_begin,
		const thrust::host_vector<int>& tLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
	clast::hit::onTargetBoundary(
			kMerLength,
			t_begin,
			tLengthArray,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
}

inline void onCorner(
		const int allowableGap,
		const int t_begin,
		const int q_begin,
		const thrust::host_vector<int>& tLengthArray,
		const thrust::host_vector<int>& qLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
	clast::hit::onCorner(
			allowableGap,
			t_begin,
			q_begin,
			tLengthArray,
			qLengthArray,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
}

#endif
