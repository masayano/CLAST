#ifndef C_DEVICE_SEQ_LIST_TARGET_CUH_
#define C_DEVICE_SEQ_LIST_TARGET_CUH_

#include "CDeviceSeqList.cuh"
#include "CHostSeqList_target.cuh"

class CDeviceSeqList_target : public CDeviceSeqList {
public:
	CDeviceSeqList_target(
			const CHostSeqList* s,
			const int startID,
			const int endID);
};

#endif
