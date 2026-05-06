#ifndef C_DEVICE_HIT_LIST_CUH_
#define C_DEVICE_HIT_LIST_CUH_

#include "host/CHostResultHolder.cuh"
#include "host/CHostSetting.cuh"
#include "device/CDeviceHashTable.cuh"
#include "device/seq/query.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

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

namespace clast::hit {

// Returns true when a result-hit row is an exact duplicate of its predecessor
// (same targetID, targetIdx, queryID, queryIdx).
struct is_duplicate {
	template <class Tuple>
	__host__ __device__ bool operator()(const Tuple& tuple) const {
		using namespace thrust;
		return (get<0>(tuple) == get<4>(tuple))
			&& (get<1>(tuple) == get<5>(tuple))
			&& (get<2>(tuple) == get<6>(tuple))
			&& (get<3>(tuple) == get<7>(tuple));
	}
};

// Removes consecutive duplicate result-hit rows in-place.
// Two rows are duplicates when they share the same (targetID, targetIdx,
// queryID, queryIdx).  The first occurrence is always kept.
template <class ThrustVectorInt, class ThrustVectorDouble>
void removeDuplicateResultHits(
		ThrustVectorInt&    targetIDArray,
		ThrustVectorInt&    targetIndexArray,
		ThrustVectorInt&    queryIDArray,
		ThrustVectorInt&    queryIndexArray,
		ThrustVectorInt&    tHitLengthArray,
		ThrustVectorInt&    qHitLengthArray,
		ThrustVectorInt&    matchNumArray,
		ThrustVectorInt&    scoreArray,
		ThrustVectorDouble& evalueArray) {
	using namespace thrust;
	if (targetIDArray.size() > 1) {
		const int new_size = remove_if(
				make_zip_iterator(
						make_tuple(
								targetIDArray   .begin() + 1,
								targetIndexArray.begin() + 1,
								queryIDArray    .begin() + 1,
								queryIndexArray .begin() + 1,
								tHitLengthArray .begin() + 1,
								qHitLengthArray .begin() + 1,
								matchNumArray   .begin() + 1,
								scoreArray      .begin() + 1,
								evalueArray     .begin() + 1
						)
				),
				make_zip_iterator(
						make_tuple(
								targetIDArray   .end(),
								targetIndexArray.end(),
								queryIDArray    .end(),
								queryIndexArray .end(),
								tHitLengthArray .end(),
								qHitLengthArray .end(),
								matchNumArray   .end(),
								scoreArray      .end(),
								evalueArray     .end()
						)
				),
				make_zip_iterator(
						make_tuple(
								targetIDArray   .begin() + 1,
								targetIndexArray.begin() + 1,
								queryIDArray    .begin() + 1,
								queryIndexArray .begin() + 1,
								targetIDArray   .begin(),
								targetIndexArray.begin(),
								queryIDArray    .begin(),
								queryIndexArray .begin()
						)
				),
				is_duplicate()
		) - make_zip_iterator(
				make_tuple(
						targetIDArray   .begin(),
						targetIndexArray.begin(),
						queryIDArray    .begin(),
						queryIndexArray .begin(),
						tHitLengthArray .begin(),
						qHitLengthArray .begin(),
						matchNumArray   .begin(),
						scoreArray      .begin(),
						evalueArray     .begin()
				)
		);

		targetIDArray   .resize(new_size);
		targetIndexArray.resize(new_size);
		queryIDArray    .resize(new_size);
		queryIndexArray .resize(new_size);
		tHitLengthArray .resize(new_size);
		qHitLengthArray .resize(new_size);
		matchNumArray   .resize(new_size);
		scoreArray      .resize(new_size);
		evalueArray     .resize(new_size);
	}
}

} // namespace clast::hit

#endif
