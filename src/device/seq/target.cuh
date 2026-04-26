#ifndef C_DEVICE_SEQ_LIST_TARGET_CUH_
#define C_DEVICE_SEQ_LIST_TARGET_CUH_

#include "device/seq/CDeviceSeqList.cuh"
#include "host/seq/target.cuh"

class CDeviceSeqList_target : public CDeviceSeqList {
public:
	CDeviceSeqList_target(
			const CHostSeqList* s,
			const int startID,
			const int endID);
};

#endif
