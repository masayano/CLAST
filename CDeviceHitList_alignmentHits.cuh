#ifndef C_DEVICE_HIT_LIST_ALIGNMENT_HITS_CUH_
#define C_DEVICE_HIT_LIST_ALIGNMENT_HITS_CUH_

#include "CHostSetting.cuh"
#include "CDeviceHashTable.cuh"
#include "CDeviceSeqList_query.cuh"

#include <thrust/device_vector.h>

void alignmentHits(
		const CHostSetting& s,
		const CDeviceHashTable& h,
		const CDeviceSeqList_query& q,
		const int t_begin,
		const int q_begin,
		thrust::device_vector<int>& targetIDArray,
		thrust::device_vector<int>& targetIndexArray,
		thrust::device_vector<int>& queryIDArray,
		thrust::device_vector<int>& queryIndexArray,
		thrust::device_vector<int>& tHitLengthArray,
		thrust::device_vector<int>& qHitLengthArray,
		thrust::device_vector<int>& matchNumArray,
		thrust::device_vector<int>& scoreArray);

struct make_alignmentSizeBackward {
	template <class Tuple>
	__device__ int operator() (const Tuple& tuple) const {
		using namespace thrust;

		const int queryLength = get<0>(tuple);
		const int queryIndex  = get<1>(tuple);
		const int qHitLength  = get<2>(tuple);

		return queryLength - (queryIndex + qHitLength);
	}
};

struct alignLengthBackward {
	template <class Tuple>
	__device__ bool operator() (const Tuple& tuple1, const Tuple& tuple2) const {
		const int alignmentSize_1 = thrust::get<8>(tuple1);
		const int alignmentSize_2 = thrust::get<8>(tuple2);
		return alignmentSize_1 > alignmentSize_2;
	}
};

struct alignLengthForward {
	template <class Tuple>
	__device__ bool operator() (const Tuple& tuple1, const Tuple& tuple2) const {
		const int qIdx_1 = thrust::get<3>(tuple1);
		const int qIdx_2 = thrust::get<3>(tuple2);
		return qIdx_1 > qIdx_2;
	}
};

#endif
