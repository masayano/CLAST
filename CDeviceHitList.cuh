#ifndef C_DEVICE_HIT_LIST_CUH_
#define C_DEVICE_HIT_LIST_CUH_

#include "CHostResultHolder.cuh"
#include "CHostSetting.cuh"
#include "CDeviceHashTable.cuh"
#include "CDeviceSeqList_query.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class CDeviceHitList {
	thrust::device_vector<int> targetIDArray;
	thrust::device_vector<int> targetIndexArray;
	thrust::device_vector<int> queryIDArray;
	thrust::device_vector<int> queryIndexArray;
	thrust::device_vector<int> tHitLengthArray;
	thrust::device_vector<int> qHitLengthArray;
	thrust::device_vector<int> matchNumArray;
	thrust::device_vector<int> scoreArray;
	thrust::device_vector<double> evalueArray;
public:
	CDeviceHitList(
			const CHostSetting& s,
			const CDeviceHashTable& h,
			const CDeviceSeqList_query& q,
			const int t_begin,
			const int q_begin);
	void getResult(CHostResultHolder& holder);
};

struct is_duplicate {
	template <class Tuple>
	__device__ bool operator() (const Tuple& tuple) const {
		using namespace thrust;
		const int targetID  = get<0>(tuple);
		const int targetIdx = get<1>(tuple);
		const int queryID   = get<2>(tuple);
		const int queryIdx  = get<3>(tuple);
		const int post_targetID  = get<4>(tuple);
		const int post_targetIdx = get<5>(tuple);
		const int post_queryID   = get<6>(tuple);
		const int post_queryIdx  = get<7>(tuple);
		return (targetID == post_targetID)
				&& (targetIdx == post_targetIdx)
				&& (queryID   == post_queryID)
				&& (queryIdx  == post_queryIdx);
	}
};

#endif
