#ifndef C_DEVICE_SEQ_LIST_QUERY_CUH_
#define C_DEVICE_SEQ_LIST_QUERY_CUH_

#include "CDeviceSeqList.cuh"
#include "CHostSeqList.cuh"
#include "CHostSetting.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class CDeviceSeqList_query : public CDeviceSeqList {
	thrust::device_vector<long> hashIndex;
public:
	CDeviceSeqList_query(
			const CHostSetting& setting,
			const CHostSeqList* s,
			const int startID,
			const int endID);
	const thrust::device_vector<long>& getHashIndex(void) const;
};

#endif
