#ifndef C_DEVICE_HIT_LIST_SEED_HOST_API_CUH_
#define C_DEVICE_HIT_LIST_SEED_HOST_API_CUH_

#include <thrust/host_vector.h>

void deleteDuplicateSeeds(
		const int allowableWidth,
		const int allowableGap,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray);

void deleteSeedHasNotNearPair(
		const int allowableWidth,
		const int allowableGap,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray);

void onQueryBoundary(
		const int kMerLength,
		const int q_begin,
		const thrust::host_vector<int> qLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray);

void onTargetBoundary(
		const int kMerLength,
		const int t_begin,
		const thrust::host_vector<int> tLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray);

void onCorner(
		const int allowableGap,
		const int t_begin,
		const int q_begin,
		const thrust::host_vector<int> tLengthArray,
		const thrust::host_vector<int> qLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray);

#endif