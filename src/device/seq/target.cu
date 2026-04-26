#include "device/seq/target.cuh"

CDeviceSeqList_target::CDeviceSeqList_target(
		const CHostSeqList* s,
		const int startID,
		const int endID)
		: CDeviceSeqList(s, startID, endID) {}
