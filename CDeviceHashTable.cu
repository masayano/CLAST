#include "CDeviceHashTable.cuh"

#ifdef MODE_TEST
	#include "CTest.cuh"
#endif /* MODE_TEST */

/******************************** non class function ***************************************/
#include "CStridedRange.cuh"

#include "SRead2bit.cuh"
#include "SRepelHasOddBase.cuh"

#include <new>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

thrust::device_vector<long> createHashIndex(
		const int lMerLength,
		const int strideLength,
		const thrust::device_vector<char>& baseArray) {
	using namespace thrust;

	/* prepare strided range */
	typedef device_vector<char>::const_iterator Iterator;
	CStridedRange<Iterator> stridedBase[lMerLength];
	for(int i = 0; i < lMerLength; ++i) {
		new(stridedBase + i) CStridedRange<Iterator>(baseArray.begin()+i, baseArray.end()+i, strideLength);
	}

	/* construct hash index */
	const int hashIndexSize = baseArray.size() / strideLength;
	device_vector<long> hashIndex(hashIndexSize);
	device_vector<char> flgArray (hashIndexSize);
	for(int i = 0; i < lMerLength; ++i) {
		thrust::transform(
				stridedBase[i].begin(),
				stridedBase[i].begin() + hashIndex.size(),
				make_zip_iterator(
						make_tuple(
								hashIndex.begin(),
								flgArray .begin()
						)
				),
				make_zip_iterator(
						make_tuple(
								hashIndex.begin(),
								flgArray .begin()
						)
				),
				read2bit()
		);
	}

	/* gather k-mer which has odd base */
	thrust::transform(
			hashIndex.begin(),
			hashIndex.end(),
			flgArray.begin(),
			hashIndex.begin(),
			repel_hasOddBase()
	);

	#ifdef MODE_TEST
	CTest::printNonoverlappedHashIndex(strideLength, baseArray, hashIndex);
	#endif /* MODE_TEST */
	return hashIndex;
}

void makeIndexArray(
		const int baseLength,
		const int strideLength,
		thrust::device_vector<int>& indexArray) {
	indexArray.resize(baseLength / strideLength);
	thrust::sequence(indexArray.begin(), indexArray.end());
	thrust::transform(
			indexArray.begin(),
			indexArray.end(),
			thrust::make_constant_iterator(strideLength),
			indexArray.begin(),
			thrust::multiplies<int>());

}

void makeGatewayKey(
		const thrust::device_vector<long>& hashIndex,
		thrust::device_vector<long>& gatewayKey) {
	gatewayKey.resize(hashIndex.size());
	gatewayKey = hashIndex;
	const int newSize = thrust::unique(gatewayKey.begin(), gatewayKey.end()) - gatewayKey.begin();
	gatewayKey.resize(newSize);
}

void makeCellSizeArray(
		const thrust::device_vector<long>& hashIndex,
		const thrust::device_vector<long>& gatewayKey,
		thrust::device_vector<int>& cellSizeArray) {
	using namespace thrust;

	cellSizeArray.resize(hashIndex.size());
	reduce_by_key(
			hashIndex.begin(),
			hashIndex.end(),
			make_constant_iterator<int>(1),
			cellSizeArray.begin(),
			cellSizeArray.begin());
	cellSizeArray.resize(gatewayKey.size());
}

void makeGatewayIndex(
		const thrust::device_vector<int>& cellSizeArray,
		thrust::device_vector<int>& gatewayIndex) {
	gatewayIndex.resize(cellSizeArray.size());
	exclusive_scan(
			cellSizeArray.begin(),
			cellSizeArray.end(),
			gatewayIndex.begin());
}

struct removeRepeat {
	__device__ int operator() (const int cellSize, const int cutRepeat) {
		return (cellSize < cutRepeat) ? cellSize : 0;
	}
};

void removeRepeatedSequence(
		const int cutRepeat,
		thrust::device_vector<int>& cellSizeArray) {
	thrust::transform(
		cellSizeArray.begin(),
		cellSizeArray.end(),
		thrust::make_constant_iterator(cutRepeat),
		cellSizeArray.begin(),
		removeRepeat());
}

/********************************** class function *****************************************/

#include <thrust/sequence.h>

CDeviceHashTable::CDeviceHashTable(
		const CHostSetting& s,
		const CHostSeqList_target& t,
		const int t_begin,
		const int t_end)
		: target(&t, t_begin, t_end) {
	thrust::device_vector<long> hashIndex = createHashIndex(
			s.getLMerLength(), 
			s.getStrideLength(),
			target.getBaseArray());

	makeIndexArray(
			target.getBaseArray().size(),
			s.getStrideLength(),
			indexArray);

	thrust::sort_by_key(hashIndex.begin(), hashIndex.end(), indexArray.begin());

	makeGatewayKey   (hashIndex, gatewayKey);
	makeCellSizeArray(hashIndex, gatewayKey, cellSizeArray);
	makeGatewayIndex (cellSizeArray, gatewayIndex);

	const int cutRepeat = s.getCutRepeat();
	if(cutRepeat != -1) {
		removeRepeatedSequence(cutRepeat, cellSizeArray);
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
