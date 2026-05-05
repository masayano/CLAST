#ifndef C_DEVICE_HIT_LIST_ALIGNMENT_HITS_CUH_
#define C_DEVICE_HIT_LIST_ALIGNMENT_HITS_CUH_

#include "host/CHostSetting.cuh"
#include "device/CDeviceHashTable.cuh"
#include "device/seq/query.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

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

namespace clast::hit {

// Computes backward alignment space: queryLength - (queryIndex + qHitLength).
struct make_alignmentSizeBackward {
	template <class Tuple>
	__host__ __device__ int operator() (const Tuple& tuple) const {
		using namespace thrust;

		const int queryLength = get<0>(tuple);
		const int queryIndex  = get<1>(tuple);
		const int qHitLength  = get<2>(tuple);

		return queryLength - (queryIndex + qHitLength);
	}
};

// Descending comparator on alignmentSize (element 8 of the sort tuple).
struct alignLengthBackward {
	template <class Tuple1, class Tuple2>
	__host__ __device__ bool operator() (const Tuple1& tuple1, const Tuple2& tuple2) const {
		const int alignmentSize_1 = thrust::get<8>(tuple1);
		const int alignmentSize_2 = thrust::get<8>(tuple2);
		return alignmentSize_1 > alignmentSize_2;
	}
};

// Descending comparator on queryIndex (element 3 of the sort tuple).
struct alignLengthForward {
	template <class Tuple1, class Tuple2>
	__host__ __device__ bool operator() (const Tuple1& tuple1, const Tuple2& tuple2) const {
		const int qIdx_1 = thrust::get<3>(tuple1);
		const int qIdx_2 = thrust::get<3>(tuple2);
		return qIdx_1 > qIdx_2;
	}
};

// Sorts seed rows in descending order of backward alignment space
// (queryLength[queryID] - (queryIndex + qHitLength)), so that the kernel
// can process rows with the largest backward extension first.
// Precondition: every queryIDArray[i] - q_begin is a valid index into qLengthArray.
template <class ThrustVectorInt>
void sortBeforeAlignBackWard(
		const int q_begin,
		const ThrustVectorInt& qLengthArray,
		ThrustVectorInt& targetIDArray,
		ThrustVectorInt& queryIDArray,
		ThrustVectorInt& targetIndexArray,
		ThrustVectorInt& queryIndexArray,
		ThrustVectorInt& tHitLengthArray,
		ThrustVectorInt& qHitLengthArray,
		ThrustVectorInt& matchNumArray,
		ThrustVectorInt& scoreArray) {
	using namespace thrust;
	ThrustVectorInt alignmentSize(targetIDArray.size());
	transform(
		make_zip_iterator(make_tuple(
			make_permutation_iterator(qLengthArray.begin() - q_begin, queryIDArray.begin()),
			queryIndexArray .begin(),
			qHitLengthArray .begin()
		)),
		make_zip_iterator(make_tuple(
			make_permutation_iterator(qLengthArray.begin() - q_begin, queryIDArray.end()),
			queryIndexArray .end(),
			qHitLengthArray .end()
		)),
		alignmentSize.begin(),
		make_alignmentSizeBackward()
	);
	sort(
		make_zip_iterator(make_tuple(
			targetIDArray   .begin(),
			queryIDArray    .begin(),
			targetIndexArray.begin(),
			queryIndexArray .begin(),
			tHitLengthArray .begin(),
			qHitLengthArray .begin(),
			matchNumArray   .begin(),
			scoreArray      .begin(),
			alignmentSize   .begin()
		)),
		make_zip_iterator(make_tuple(
			targetIDArray   .end(),
			queryIDArray    .end(),
			targetIndexArray.end(),
			queryIndexArray .end(),
			tHitLengthArray .end(),
			qHitLengthArray .end(),
			matchNumArray   .end(),
			scoreArray      .end(),
			alignmentSize   .end()
		)),
		alignLengthBackward()
	);
}

// Sorts seed rows in descending order of queryIndex, so that the forward
// alignment kernel processes rows with the largest query start position first.
template <class ThrustVectorInt>
void sortBeforeAlignForward(
		ThrustVectorInt& targetIDArray,
		ThrustVectorInt& queryIDArray,
		ThrustVectorInt& targetIndexArray,
		ThrustVectorInt& queryIndexArray,
		ThrustVectorInt& tHitLengthArray,
		ThrustVectorInt& qHitLengthArray,
		ThrustVectorInt& matchNumArray,
		ThrustVectorInt& scoreArray) {
	using namespace thrust;
	sort(
		make_zip_iterator(make_tuple(
			targetIDArray   .begin(),
			queryIDArray    .begin(),
			targetIndexArray.begin(),
			queryIndexArray .begin(),
			tHitLengthArray .begin(),
			qHitLengthArray .begin(),
			matchNumArray   .begin(),
			scoreArray      .begin()
		)),
		make_zip_iterator(make_tuple(
			targetIDArray   .end(),
			queryIDArray    .end(),
			targetIndexArray.end(),
			queryIndexArray .end(),
			tHitLengthArray .end(),
			qHitLengthArray .end(),
			matchNumArray   .end(),
			scoreArray      .end()
		)),
		alignLengthForward()
	);
}

} // namespace clast::hit

#endif
