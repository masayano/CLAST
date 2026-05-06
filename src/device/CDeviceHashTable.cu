#include "device/CDeviceHashTable.cuh"

#include <thrust/sort.h>

/********************************** class function *****************************************/

CDeviceHashTable::CDeviceHashTable(
		const CHostSetting& s,
		const CHostSeqList_target& t,
		const int t_begin,
		const int t_end)
		: target(&t, t_begin, t_end) {
	thrust::device_vector<long> hashIndex;
	clast::hash::createHashIndex(
			s.getLMerLength(),
			s.getStrideLength(),
			target.getBaseArray(),
			hashIndex);
	#ifdef MODE_TEST
	CTest::printNonoverlappedHashIndex(s.getStrideLength(), target.getBaseArray(), hashIndex);
	#endif /* MODE_TEST */

	clast::hash::makeIndexArray(
			target.getBaseArray().size(),
			s.getStrideLength(),
			indexArray);

	thrust::sort_by_key(hashIndex.begin(), hashIndex.end(), indexArray.begin());

	clast::hash::makeGatewayKey   (hashIndex, gatewayKey);
	clast::hash::makeCellSizeArray(hashIndex, gatewayKey, cellSizeArray);
	clast::hash::makeGatewayIndex (cellSizeArray, gatewayIndex);

	const int cutRepeat = s.getCutRepeat();
	if(cutRepeat != -1) {
		clast::hash::removeRepeatedSequence(cutRepeat, cellSizeArray);
	}

	/* erase bad data from hash table */
	if(gatewayKey[0] == -1) {
		cellSizeArray[0] = 0;
	}
}

const CDeviceSeqList_target&  CDeviceHashTable::getTarget (void) const {
	return target;
}

const thrust::device_vector<long>&  CDeviceHashTable::getGatewayKey(void) const {
	return gatewayKey;
}

const thrust::device_vector<int>&  CDeviceHashTable::getCellSizeArray(void) const {
	return cellSizeArray;
}

const thrust::device_vector<int>&  CDeviceHashTable::getGatewayIndex (void) const {
	return gatewayIndex;
}

const thrust::device_vector<int>&  CDeviceHashTable::getIndexArray   (void) const {
	return indexArray;
}
