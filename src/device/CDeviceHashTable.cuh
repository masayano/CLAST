#ifndef C_DEVICE_HASH_TABLE_CUH_
#define C_DEVICE_HASH_TABLE_CUH_

#include "host/CHostSetting.cuh"
#include "host/seq/query.cuh"
#include "device/seq/target.cuh"
#include "util/CStridedRange.cuh"
#include "util/SRead2bit.cuh"
#include "util/SRepelHasOddBase.cuh"

#include <new>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

class CDeviceHashTable {
	const CDeviceSeqList_target target;
	thrust::device_vector<long> gatewayKey;
	thrust::device_vector<int> cellSizeArray;
	thrust::device_vector<int> gatewayIndex;
	thrust::device_vector<int> indexArray;
public:
	CDeviceHashTable(
			const CHostSetting& s,
			const CHostSeqList_target& t,
			const int t_begin,
			const int t_end);
	const CDeviceSeqList_target& getTarget(void) const;
	const thrust::device_vector<long>& getGatewayKey(void) const;
	const thrust::device_vector<int>& getCellSizeArray(void) const;
	const thrust::device_vector<int>& getGatewayIndex (void) const;
	const thrust::device_vector<int>& getIndexArray   (void) const;
};

namespace clast::hash {

template <class ThrustVectorChar, class ThrustVectorLong>
void createHashIndex(
		const int lMerLength,
		const int strideLength,
		const ThrustVectorChar& baseArray,
		ThrustVectorLong& hashIndex) {
	using namespace thrust;
	typedef typename ThrustVectorChar::const_iterator Iterator;

	CStridedRange<Iterator> stridedBase[lMerLength];
	for(int i = 0; i < lMerLength; ++i) {
		new(stridedBase + i) CStridedRange<Iterator>(baseArray.begin()+i, baseArray.end()+i, strideLength);
	}

	const int hashIndexSize = static_cast<int>(baseArray.size()) / strideLength;
	hashIndex.assign(hashIndexSize, 0L);
	ThrustVectorChar flgArray(hashIndexSize, 0);

	for(int i = 0; i < lMerLength; ++i) {
		thrust::transform(
				stridedBase[i].begin(),
				stridedBase[i].begin() + hashIndex.size(),
				make_zip_iterator(make_tuple(hashIndex.begin(), flgArray.begin())),
				make_zip_iterator(make_tuple(hashIndex.begin(), flgArray.begin())),
				read2bit()
		);
	}

	thrust::transform(
			hashIndex.begin(),
			hashIndex.end(),
			flgArray.begin(),
			hashIndex.begin(),
			repel_hasOddBase()
	);
}

// Zeros out a cell size when it equals or exceeds the repeat cutoff,
// suppressing highly repeated k-mers from alignment.
struct removeRepeat {
	__host__ __device__ int operator()(const int cellSize, const int cutRepeat) const {
		return (cellSize < cutRepeat) ? cellSize : 0;
	}
};

// Produces the byte-offset position of each stride window within the sequence.
// Result: {0, stride, 2*stride, ..., (n-1)*stride} where n = baseLength/strideLength.
template <class ThrustVectorInt>
void makeIndexArray(
		const int baseLength,
		const int strideLength,
		ThrustVectorInt& indexArray) {
	indexArray.resize(baseLength / strideLength);
	thrust::sequence(indexArray.begin(), indexArray.end());
	thrust::transform(
			indexArray.begin(),
			indexArray.end(),
			thrust::make_constant_iterator(strideLength),
			indexArray.begin(),
			thrust::multiplies<int>());
}

// Extracts the sorted unique hash values from a sorted hashIndex array.
// Precondition: hashIndex is sorted.
template <class ThrustVectorLong>
void makeGatewayKey(
		const ThrustVectorLong& hashIndex,
		ThrustVectorLong& gatewayKey) {
	gatewayKey.resize(hashIndex.size());
	gatewayKey = hashIndex;
	const int newSize = thrust::unique(gatewayKey.begin(), gatewayKey.end())
	                  - gatewayKey.begin();
	gatewayKey.resize(newSize);
}

// Counts how many entries in the sorted hashIndex belong to each unique key.
// Result size equals gatewayKey.size().
// Precondition: hashIndex is sorted and gatewayKey == unique keys of hashIndex.
template <class ThrustVectorLong, class ThrustVectorInt>
void makeCellSizeArray(
		const ThrustVectorLong& hashIndex,
		const ThrustVectorLong& gatewayKey,
		ThrustVectorInt& cellSizeArray) {
	cellSizeArray.resize(hashIndex.size());
	thrust::reduce_by_key(
			hashIndex.begin(),
			hashIndex.end(),
			thrust::make_constant_iterator<int>(1),
			cellSizeArray.begin(),
			cellSizeArray.begin());
	cellSizeArray.resize(gatewayKey.size());
}

// Computes the start index of each hash bucket via exclusive prefix scan of cellSizeArray.
template <class ThrustVectorInt>
void makeGatewayIndex(
		const ThrustVectorInt& cellSizeArray,
		ThrustVectorInt& gatewayIndex) {
	gatewayIndex.resize(cellSizeArray.size());
	thrust::exclusive_scan(
			cellSizeArray.begin(),
			cellSizeArray.end(),
			gatewayIndex.begin());
}

// Zeros out entries in cellSizeArray where the cell size >= cutRepeat,
// suppressing highly repeated k-mers from alignment.
template <class ThrustVectorInt>
void removeRepeatedSequence(
		const int cutRepeat,
		ThrustVectorInt& cellSizeArray) {
	thrust::transform(
			cellSizeArray.begin(),
			cellSizeArray.end(),
			thrust::make_constant_iterator(cutRepeat),
			cellSizeArray.begin(),
			removeRepeat());
}

} // namespace clast::hash

#endif /* C_HASH_TABLE_CUH_ */
