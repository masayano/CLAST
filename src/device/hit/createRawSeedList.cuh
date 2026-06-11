#ifndef C_DEVICE_HIT_LIST_CREATE_RAW_SEED_LIST_CUH_
#define C_DEVICE_HIT_LIST_CREATE_RAW_SEED_LIST_CUH_

#include "host/CHostSetting.cuh"
#include "device/CDeviceHashTable.cuh"
#include "device/seq/query.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform.h>

void createRawSeedList(
		const CHostSetting& s,
		const CDeviceHashTable& h,
		const CDeviceSeqList_query& q,
		const int t_begin,
		const int q_begin,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray);

namespace clast::hit {

// Looks up a gateway index by key; returns 0 when key is -1 (no hash hit).
struct fillGatewayIndex {
	__host__ __device__ int operator()(const int gatewayIndex, const int key) const {
		return key == -1 ? 0 : gatewayIndex;
	}
};

// Looks up a cell size by key; returns 0 when key is -1 (no hash hit).
struct fillCellSize {
	__host__ __device__ int operator()(const int cellSize, const int key) const {
		return key == -1 ? 0 : cellSize;
	}
};

// Maps key == -1 (no hash hit) to 0 so it can be used as a permutation index
// without reading out of bounds. fillGatewayIndex/fillCellSize discard the
// looked-up value whenever the original key is -1 anyway.
struct clampMissingKey {
	__host__ __device__ int operator()(const int key) const {
		return key < 0 ? 0 : key;
	}
};

// For each entry in gatewayKeyArray, writes hashGatewayIndexArray[key] into
// gatewayIndexArray, or 0 when key is -1.
// Precondition: every key >= 0 is a valid index into hashGatewayIndexArray.
template <class ThrustVectorInt>
void writeGatewayIndexValues(
		const ThrustVectorInt& hashGatewayIndexArray,
		const ThrustVectorInt& gatewayKeyArray,
		ThrustVectorInt& gatewayIndexArray) {
	ThrustVectorInt safeKeyArray(gatewayKeyArray.size());
	thrust::transform(
		gatewayKeyArray.begin(), gatewayKeyArray.end(),
		safeKeyArray.begin(),
		clampMissingKey());

	thrust::transform(
		thrust::make_permutation_iterator(
			hashGatewayIndexArray.begin(), safeKeyArray.begin()),
		thrust::make_permutation_iterator(
			hashGatewayIndexArray.begin(), safeKeyArray.end()),
		gatewayKeyArray.begin(),
		gatewayIndexArray.begin(),
		fillGatewayIndex());
}

// For each entry in gatewayKeyArray, writes hashCellSizeArray[key] into
// cellSizeArray, or 0 when key is -1.
// Precondition: every key >= 0 is a valid index into hashCellSizeArray.
template <class ThrustVectorInt>
void writeCellSizeValues(
		const ThrustVectorInt& hashCellSizeArray,
		const ThrustVectorInt& gatewayKeyArray,
		ThrustVectorInt& cellSizeArray) {
	ThrustVectorInt safeKeyArray(gatewayKeyArray.size());
	thrust::transform(
		gatewayKeyArray.begin(), gatewayKeyArray.end(),
		safeKeyArray.begin(),
		clampMissingKey());

	thrust::transform(
		thrust::make_permutation_iterator(
			hashCellSizeArray.begin(), safeKeyArray.begin()),
		thrust::make_permutation_iterator(
			hashCellSizeArray.begin(), safeKeyArray.end()),
		gatewayKeyArray.begin(),
		cellSizeArray.begin(),
		fillCellSize());
}

} // namespace clast::hit

#endif
