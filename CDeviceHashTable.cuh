#ifndef C_DEVICE_HASH_TABLE_CUH_
#define C_DEVICE_HASH_TABLE_CUH_

#include "CHostSetting.cuh"
#include "CHostSeqList_query.cuh"
#include "CDeviceSeqList_target.cuh"

#include <thrust/device_vector.h>

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

#endif /* C_HASH_TABLE_CUH_ */
