#include "CDeviceSeqList_query.cuh"

#include "common.hpp"

#ifdef MODE_TEST
	#include "CTest.cuh"
#endif

/********************************** non class function **********************************/
#include "SRead2bit.cuh"
#include "SRepelHasOddBase.cuh"

#include <thrust/iterator/constant_iterator.h>

void createHashIndex(
		const int lMerLength,
		const thrust::device_vector<char>& base,
		thrust::device_vector<long>& hashIndex) {
	using namespace thrust;

	/* construct hash index */
	const int hashIndexSize = base.size() - (lMerLength - 1);
	hashIndex.resize(hashIndexSize);
	device_vector<char> flgArray (hashIndexSize);
	for(int i = 0; i < lMerLength; ++i) {
		thrust::transform(
				base.begin() + i,
				base.begin() + i + hashIndex.size(),
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
	CTest::printOverlappedHashIndex(base, hashIndex);
	#endif /* MODE_TEST */
}

/********************************** class function **********************************/

CDeviceSeqList_query::CDeviceSeqList_query(
		const CHostSetting& setting,
		const CHostSeqList* s,
		const int startID,
		const int endID)
		: CDeviceSeqList(s, startID * 2, endID * 2) { // this magic number 2 = number of query strand types ('+', '-')
	createHashIndex(
			setting.getLMerLength(),
			baseArray,
			hashIndex);
}

const thrust::device_vector<long>& CDeviceSeqList_query::getHashIndex(void) const {
	return hashIndex;
}
