#include "CDeviceHitList_alignmentHits.cuh"

#include "common.hpp"
#include "krnlAlignment.cuh"
#include "CTest.cuh"

/******************************* private ************************************/
#include <thrust/sort.h>

void sortBeforeAlignBackWard(
		const int q_begin,
		const thrust::device_vector<int>& qLengthArray,
		thrust::device_vector<int>& targetIDArray,
		thrust::device_vector<int>& queryIDArray,
		thrust::device_vector<int>& targetIndexArray,
		thrust::device_vector<int>& queryIndexArray,
		thrust::device_vector<int>& tHitLengthArray,
		thrust::device_vector<int>& qHitLengthArray,
		thrust::device_vector<int>& matchNumArray,
		thrust::device_vector<int>& scoreArray) {
	using namespace thrust;

	device_vector<int> alignmentSize(targetIDArray.size());
	thrust::transform(
			make_zip_iterator(
					make_tuple(
							make_permutation_iterator(
									qLengthArray.begin() - q_begin,
									queryIDArray.begin()
							),
							queryIndexArray.begin(),
							qHitLengthArray.begin()
					)
			),
			make_zip_iterator(
					make_tuple(
							make_permutation_iterator(
									qLengthArray.begin() - q_begin,
									queryIDArray.end()
							),
							queryIndexArray.end(),
							qHitLengthArray.end()
					)
			),
			alignmentSize.begin(),
			make_alignmentSizeBackward()
	);

	sort(
			make_zip_iterator(
					make_tuple(
							targetIDArray   .begin(),
							queryIDArray    .begin(),
							targetIndexArray.begin(),
							queryIndexArray .begin(),
							tHitLengthArray .begin(),
							qHitLengthArray .begin(),
							matchNumArray   .begin(),
							scoreArray      .begin(),
							alignmentSize   .begin()
					)
			),
			make_zip_iterator(
					make_tuple(
							targetIDArray   .end(),
							queryIDArray    .end(),
							targetIndexArray.end(),
							queryIndexArray .end(),
							tHitLengthArray .end(),
							qHitLengthArray .end(),
							matchNumArray   .end(),
							scoreArray      .end(),
							alignmentSize   .end()
					)
			),
			alignLengthBackward()
	);
}

void sortBeforeAlignForward(
		thrust::device_vector<int>& targetIDArray,
		thrust::device_vector<int>& queryIDArray,
		thrust::device_vector<int>& targetIndexArray,
		thrust::device_vector<int>& queryIndexArray,
		thrust::device_vector<int>& tHitLengthArray,
		thrust::device_vector<int>& qHitLengthArray,
		thrust::device_vector<int>& matchNumArray,
		thrust::device_vector<int>& scoreArray) {
	using namespace thrust;

	sort(
			make_zip_iterator(
					make_tuple(
							targetIDArray   .begin(),
							queryIDArray    .begin(),
							targetIndexArray.begin(),
							queryIndexArray .begin(),
							tHitLengthArray .begin(),
							qHitLengthArray .begin(),
							matchNumArray   .begin(),
							scoreArray      .begin()
					)
			),
			make_zip_iterator(
					make_tuple(
							targetIDArray   .end(),
							queryIDArray    .end(),
							targetIndexArray.end(),
							queryIndexArray .end(),
							tHitLengthArray .end(),
							qHitLengthArray .end(),
							matchNumArray   .end(),
							scoreArray      .end()
					)
			),
			alignLengthForward()
	);
}

/******************************** public ************************************/

void alignmentHits(
		const CHostSetting& s,
		const CDeviceHashTable& h,
		const CDeviceSeqList_query& q,
		const int t_begin,
		const int q_begin,
		thrust::device_vector<int>& targetIDArray,
		thrust::device_vector<int>& targetIndexArray,
		thrust::device_vector<int>& queryIDArray,
		thrust::device_vector<int>& queryIndexArray,
		thrust::device_vector<int>& tHitLengthArray,
		thrust::device_vector<int>& qHitLengthArray,
		thrust::device_vector<int>& matchNumArray,
		thrust::device_vector<int>& scoreArray) {
        #ifdef TIME_ATTACK
                float elapsed_time_ms=0.0f;
                cudaEvent_t start, stop;
                cudaEventCreate( &start );
                cudaEventCreate( &stop  );
                cudaEventRecord( start, 0 );
                std::cout << "  ...allignment seeds";
        #endif /* TIME_ATTACK */
	const int blockDim_x = 32;
	const int  hitNum = targetIDArray.size();
	if(s.getFlgLocal()) {
		sortBeforeAlignBackWard(
				q_begin,
				q.getLengthArray(),
				targetIDArray,
				queryIDArray,
				targetIndexArray,
				queryIndexArray,
				tHitLengthArray,
				qHitLengthArray,
				matchNumArray,
				scoreArray);
		const int allowableGap = s.getAllowableGap();
		const int tempNodeWidth = 1 + 2 * (allowableGap + MARGIN);
		const int tempNodeArraySize = hitNum * tempNodeWidth;
		thrust::device_vector<int> tempNodeArray_score     (tempNodeArraySize);
		thrust::device_vector<int> tempNodeArray_vertical  (tempNodeArraySize);
		thrust::device_vector<int> tempNodeArray_horizontal(tempNodeArraySize);
		thrust::device_vector<int> tempNodeArray_matchNum  (tempNodeArraySize);
		const int blockNum = (tempNodeArraySize / 256) + 1;
		const dim3 initTempNodeBlock(65535, (blockNum/65535)+1, 1);
		initTempNodeArray<<<initTempNodeBlock, 256>>>(
				hitNum,
				allowableGap,
				raw_pointer_cast( &*tempNodeArray_score     .begin() ),
				raw_pointer_cast( &*tempNodeArray_vertical  .begin() ),
				raw_pointer_cast( &*tempNodeArray_horizontal.begin() ),
				raw_pointer_cast( &*tempNodeArray_matchNum  .begin() )
		);
		localAlignBackward<<<(hitNum/blockDim_x)+1, blockDim_x>>>(
				hitNum,
				allowableGap,
				t_begin,
				q_begin,
				raw_pointer_cast( &*h.getTarget().getGateway()    .begin() ),
				raw_pointer_cast( &*h.getTarget().getLengthArray().begin() ),
				raw_pointer_cast( &*h.getTarget().getBaseArray()  .begin() ),
				raw_pointer_cast( &*q.getGateway()    .begin() ),
				raw_pointer_cast( &*q.getLengthArray().begin() ),
				raw_pointer_cast( &*q.getBaseArray()  .begin() ),
				raw_pointer_cast( &*targetIDArray   .begin() ),
				raw_pointer_cast( &*queryIDArray    .begin() ),
				raw_pointer_cast( &*targetIndexArray.begin() ),
				raw_pointer_cast( &*queryIndexArray .begin() ),
				raw_pointer_cast( &*tHitLengthArray .begin() ),
				raw_pointer_cast( &*qHitLengthArray .begin() ),
				raw_pointer_cast( &*matchNumArray   .begin() ),
				raw_pointer_cast( &*scoreArray      .begin() ),
				raw_pointer_cast( &*tempNodeArray_score     .begin() ),
				raw_pointer_cast( &*tempNodeArray_vertical  .begin() ),
				raw_pointer_cast( &*tempNodeArray_horizontal.begin() ),
				raw_pointer_cast( &*tempNodeArray_matchNum  .begin() )
		);
		sortBeforeAlignForward(
				targetIDArray,
				queryIDArray,
				targetIndexArray,
				queryIndexArray,
				tHitLengthArray,
				qHitLengthArray,
				matchNumArray,
				scoreArray);
                initTempNodeArray<<<initTempNodeBlock, 256>>>(
                                hitNum,
                                allowableGap,
                                raw_pointer_cast( &*tempNodeArray_score     .begin() ),
                                raw_pointer_cast( &*tempNodeArray_vertical  .begin() ),
                                raw_pointer_cast( &*tempNodeArray_horizontal.begin() ),
                                raw_pointer_cast( &*tempNodeArray_matchNum  .begin() )
                );
		localAlignForward<<<(hitNum/blockDim_x)+1, blockDim_x>>>(
				hitNum,
				allowableGap,
				t_begin,
				q_begin,
				raw_pointer_cast( &*h.getTarget().getGateway()    .begin() ),
				raw_pointer_cast( &*h.getTarget().getLengthArray().begin() ),
				raw_pointer_cast( &*h.getTarget().getBaseArray()  .begin() ),
				raw_pointer_cast( &*q.getGateway()    .begin() ),
				raw_pointer_cast( &*q.getLengthArray().begin() ),
				raw_pointer_cast( &*q.getBaseArray()  .begin() ),
				raw_pointer_cast( &*targetIDArray   .begin() ),
				raw_pointer_cast( &*queryIDArray    .begin() ),
				raw_pointer_cast( &*targetIndexArray.begin() ),
				raw_pointer_cast( &*queryIndexArray .begin() ),
				raw_pointer_cast( &*tHitLengthArray .begin() ),
				raw_pointer_cast( &*qHitLengthArray .begin() ),
				raw_pointer_cast( &*matchNumArray   .begin() ),
				raw_pointer_cast( &*scoreArray      .begin() ),
				raw_pointer_cast( &*tempNodeArray_score     .begin() ),
				raw_pointer_cast( &*tempNodeArray_vertical  .begin() ),
				raw_pointer_cast( &*tempNodeArray_horizontal.begin() ),
				raw_pointer_cast( &*tempNodeArray_matchNum  .begin() )
		);
	} else {
		sortBeforeAlignBackWard(
				q_begin,
				q.getLengthArray(),
				targetIDArray,
				queryIDArray,
				targetIndexArray,
				queryIndexArray,
				tHitLengthArray,
				qHitLengthArray,
				matchNumArray,
				scoreArray);
		const int allowableGap = s.getAllowableGap();
		const int tempNodeWidth = 1 + 2 * (allowableGap + MARGIN);
                const int tempNodeArraySize = hitNum * tempNodeWidth;
                thrust::device_vector<int> tempNodeArray_score     (tempNodeArraySize);
                thrust::device_vector<int> tempNodeArray_vertical  (tempNodeArraySize);
                thrust::device_vector<int> tempNodeArray_horizontal(tempNodeArraySize);
                thrust::device_vector<int> tempNodeArray_matchNum  (tempNodeArraySize);
                const int blockNum = (tempNodeArraySize / 256) + 1;
                const dim3 initTempNodeBlock(65535, (blockNum/65535)+1, 1);
		initTempNodeArray<<<initTempNodeBlock, 256>>>(
				hitNum,
				allowableGap,
				raw_pointer_cast( &*tempNodeArray_score     .begin() ),
				raw_pointer_cast( &*tempNodeArray_vertical  .begin() ),
				raw_pointer_cast( &*tempNodeArray_horizontal.begin() ),
				raw_pointer_cast( &*tempNodeArray_matchNum  .begin() )
		);
		globalAlignBackward<<<(hitNum/blockDim_x)+1, blockDim_x>>>(
				hitNum,
				allowableGap,
				t_begin,
				q_begin,
				raw_pointer_cast( &*h.getTarget().getGateway()    .begin() ),
				raw_pointer_cast( &*h.getTarget().getLengthArray().begin() ),
				raw_pointer_cast( &*h.getTarget().getBaseArray()  .begin() ),
				raw_pointer_cast( &*q.getGateway()    .begin() ),
				raw_pointer_cast( &*q.getLengthArray().begin() ),
				raw_pointer_cast( &*q.getBaseArray()  .begin() ),
				raw_pointer_cast( &*targetIDArray   .begin() ),
				raw_pointer_cast( &*queryIDArray    .begin() ),
				raw_pointer_cast( &*targetIndexArray.begin() ),
				raw_pointer_cast( &*queryIndexArray .begin() ),
				raw_pointer_cast( &*tHitLengthArray .begin() ),
				raw_pointer_cast( &*qHitLengthArray .begin() ),
				raw_pointer_cast( &*matchNumArray   .begin() ),
				raw_pointer_cast( &*scoreArray      .begin() ),
				raw_pointer_cast( &*tempNodeArray_score     .begin() ),
				raw_pointer_cast( &*tempNodeArray_vertical  .begin() ),
				raw_pointer_cast( &*tempNodeArray_horizontal.begin() ),
				raw_pointer_cast( &*tempNodeArray_matchNum  .begin() )
		);
		sortBeforeAlignForward(
				targetIDArray,
				queryIDArray,
				targetIndexArray,
				queryIndexArray,
				tHitLengthArray,
				qHitLengthArray,
				matchNumArray,
				scoreArray);
		initTempNodeArray<<<initTempNodeBlock, 256>>>(
				hitNum,
				allowableGap,
				raw_pointer_cast( &*tempNodeArray_score     .begin() ),
				raw_pointer_cast( &*tempNodeArray_vertical  .begin() ),
				raw_pointer_cast( &*tempNodeArray_horizontal.begin() ),
				raw_pointer_cast( &*tempNodeArray_matchNum  .begin() )
		);
		globalAlignForward<<<(hitNum/blockDim_x)+1, blockDim_x>>>(
				hitNum,
				allowableGap,
				t_begin,
				q_begin,
				raw_pointer_cast( &*h.getTarget().getGateway()    .begin() ),
				raw_pointer_cast( &*h.getTarget().getLengthArray().begin() ),
				raw_pointer_cast( &*h.getTarget().getBaseArray()  .begin() ),
				raw_pointer_cast( &*q.getGateway()    .begin() ),
				raw_pointer_cast( &*q.getLengthArray().begin() ),
				raw_pointer_cast( &*q.getBaseArray()  .begin() ),
				raw_pointer_cast( &*targetIDArray   .begin() ),
				raw_pointer_cast( &*queryIDArray    .begin() ),
				raw_pointer_cast( &*targetIndexArray.begin() ),
				raw_pointer_cast( &*queryIndexArray .begin() ),
				raw_pointer_cast( &*tHitLengthArray .begin() ),
				raw_pointer_cast( &*qHitLengthArray .begin() ),
				raw_pointer_cast( &*matchNumArray   .begin() ),
				raw_pointer_cast( &*scoreArray      .begin() ),
				raw_pointer_cast( &*tempNodeArray_score     .begin() ),
				raw_pointer_cast( &*tempNodeArray_vertical  .begin() ),
				raw_pointer_cast( &*tempNodeArray_horizontal.begin() ),
				raw_pointer_cast( &*tempNodeArray_matchNum  .begin() )
		);
	}
        #ifdef TIME_ATTACK
                std::cout << "......................................finished.";
                cudaEventRecord( stop, 0 );
                cudaEventSynchronize( stop );
                cudaEventElapsedTime( &elapsed_time_ms, start, stop );
                std::cout
                                << " (costs " << elapsed_time_ms << "ms) "
                                << targetIDArray.size() << " hits found."
                                << std::endl;
        #endif /* TIME_ATTACK */
	#ifdef MODE_TEST
		CTest::printIsolatedHit(
				targetIDArray,
				targetIndexArray,
				queryIDArray,
				queryIndexArray,
				hitLengthArray);
	#endif /* MODE_TEST */
}
