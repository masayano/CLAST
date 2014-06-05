#include "CDeviceSeqList.cuh"

#include "common.hpp"

#ifdef MODE_TEST
#include "CTest.cuh"
#endif /* MODE_TEST */

CDeviceSeqList::CDeviceSeqList(
		const CHostSeqList* s,
		const int startID,
		const int endID)
		: indexArray (s->getIndexArray (startID, endID)),
		  IDArray    (s->getIDArray    (startID, endID)),
		  baseArray  (s->getBaseArray  (startID, endID)),
		  lengthArray(s->getLengthArray(startID, endID)),
		  gateway    (s->getGateway    (startID, endID)) {
	#ifdef MODE_TEST
	CTest::printAddedSeq(this);
	#endif /* MODE_TEST */
}

const thrust::device_vector<int>&  CDeviceSeqList::getIndexArray (void) const {
	return indexArray;
}

const thrust::device_vector<int>&  CDeviceSeqList::getIDArray    (void) const {
	return IDArray;
}

const thrust::device_vector<char>& CDeviceSeqList::getBaseArray (void) const {
	return baseArray;
}

const thrust::device_vector<int>&  CDeviceSeqList::getLengthArray(void) const {
	return lengthArray;
}

const thrust::device_vector<int>&  CDeviceSeqList::getGateway    (void) const {
	return gateway;
}
