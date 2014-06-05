#ifndef C_DEVICE_HIT_LIST_DELETE_SEEDS_ON_SEQUENCE_BOUNDARY_CUH_
#define C_DEVICE_HIT_LIST_DELETE_SEEDS_ON_SEQUENCE_BOUNDARY_CUH_

#include "CHostSetting.cuh"
#include "CDeviceHashTable.cuh"
#include "CDeviceSeqList_query.cuh"

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

struct is_onBoundary {
	template <class Tuple>
	__host__ bool operator() (const Tuple& tuple) const {
		const int kMerLength = thrust::get<0>(tuple);
		const int length     = thrust::get<1>(tuple);
		const int idx        = thrust::get<2>(tuple);
		return (kMerLength + idx > length);
	}
};

struct is_onCorner {
	template <class Tuple>
	__host__ bool operator() (const Tuple& tuple) const {
		using namespace thrust;
		const int allowableGap = get<0>(tuple);
		const int tLength      = get<1>(tuple);
		const int tIdx         = get<2>(tuple);
		const int qLength      = get<3>(tuple);
		const int qIdx         = get<4>(tuple);
		const int tHitStartIdx = tIdx - qIdx;
		if(tHitStartIdx + allowableGap < 0) {
			return true;
		} else if(tHitStartIdx + qLength - allowableGap > tLength) {
			return true;
		} else {
			return false;
		}
	}
};

#endif
