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

// Returns gatewayKey when flg is true, -1 otherwise.
struct modifyGatewayKey {
	__host__ __device__ int operator()(const int gatewayKey, const bool flg) const {
		return flg ? gatewayKey : -1;
	}
};

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

// For each entry in gatewayKeyArray, writes hashGatewayIndexArray[key] into
// gatewayIndexArray, or 0 when key is -1.
// Precondition: every key >= 0 is a valid index into hashGatewayIndexArray.
template <class ThrustVectorInt>
void writeGatewayIndexValues(
		const ThrustVectorInt& hashGatewayIndexArray,
		const ThrustVectorInt& gatewayKeyArray,
		ThrustVectorInt& gatewayIndexArray) {
	thrust::transform(
		thrust::make_permutation_iterator(
			hashGatewayIndexArray.begin(), gatewayKeyArray.begin()),
		thrust::make_permutation_iterator(
			hashGatewayIndexArray.begin(), gatewayKeyArray.end()),
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
	thrust::transform(
		thrust::make_permutation_iterator(
			hashCellSizeArray.begin(), gatewayKeyArray.begin()),
		thrust::make_permutation_iterator(
			hashCellSizeArray.begin(), gatewayKeyArray.end()),
		gatewayKeyArray.begin(),
		cellSizeArray.begin(),
		fillCellSize());
}

} // namespace clast::hit

#endif
