#ifndef C_DEVICE_SEQ_LIST_CUH_
#define C_DEVICE_SEQ_LIST_CUH_

#include "CHostSeqList.cuh"

#include <thrust/device_vector.h>

class CDeviceSeqList {
protected:
	const thrust::device_vector<int>  indexArray;
	const thrust::device_vector<int>  IDArray;
	const thrust::device_vector<char> baseArray;
	const thrust::device_vector<int>  lengthArray;
	const thrust::device_vector<int>  gateway;
public:
	CDeviceSeqList(const CHostSeqList* s, const int startID, const int endID);
	const thrust::device_vector<int>&  getIndexArray (void) const;
	const thrust::device_vector<int>&  getIDArray    (void) const;
	const thrust::device_vector<char>& getBaseArray  (void) const;
	const thrust::device_vector<int>&  getLengthArray(void) const;
	const thrust::device_vector<int>&  getGateway    (void) const;
};

#endif /* C_DEVUCE_SEQUENCE_LIST_CUH_ */
