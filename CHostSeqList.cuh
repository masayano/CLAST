#ifndef C_HOST_SEQ_LIST_CUH_
#define C_HOST_SEQ_LIST_CUH_

#include "CHostSetting.cuh"
#include "CHostFASTA.hpp"

#include <string>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class CHostSeqList {
protected:
	std::vector<std::string> labelArray;
	thrust::host_vector<int>  indexArray;	// Index of intra-sequence
	thrust::host_vector<int>  IDArray;		// Sequence ID
	thrust::host_vector<char> baseArray;	// Base
	thrust::host_vector<int>  lengthArray;	// Length of each sequences
	thrust::host_vector<int>  gateway;		// Gateway array
public:
	const std::string& getLabel(const int seqID) const;
	const thrust::device_vector<int>  getIndexArray (const int startID, const int endID) const;
	const thrust::device_vector<int>  getIDArray    (const int startID, const int endID) const;
	const thrust::device_vector<char> getBaseArray  (const int startID, const int endID) const;
	const thrust::device_vector<int>  getLengthArray(const int startID, const int endID) const;
	const thrust::device_vector<int>  getGateway    (const int startID, const int endID) const;
	int getGatewayIdx (const int seqID) const;
	int getGatewaySize(void) const;
};

#endif /* C_SEQUENCE_LIST_CUH_ */
