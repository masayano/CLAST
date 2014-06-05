#include "CDeviceHitList.cuh"

#include "CDeviceHitList_alignmentHits.cuh"
#include "CDeviceHitList_createRawSeedList.cuh"
#include "CDeviceHitList_deleteBadHits.cuh"
#include "CDeviceHitList_deleteDuplicateSeeds.cuh"
#include "CDeviceHitList_deleteIsolateSeeds.cuh"
#include "CDeviceHitList_deleteSeedsOnSequenceBoundary.cuh"
#include "CDeviceHitList_sortSeeds.cuh"
#include "utilResultSorting.cuh"

#include "krnlCalculateEvalue.cuh"
#include "krnlAlignment.cuh"

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include "common.hpp"

#ifdef MODE_TEST
#include "CTest.cuh"
#endif /* MODE_TEST */

/*************************************** private **************************************/

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

void deleteHits_duplicateResult(
		thrust::device_vector<int>& targetIDArray,
		thrust::device_vector<int>& targetIndexArray,
		thrust::device_vector<int>& queryIDArray,
		thrust::device_vector<int>& queryIndexArray,
		thrust::device_vector<int>& tHitLengthArray,
		thrust::device_vector<int>& qHitLengthArray,
		thrust::device_vector<int>& matchNumArray,
		thrust::device_vector<int>& scoreArray,
		thrust::device_vector<double>& evalueArray) {
        #ifdef TIME_ATTACK
                float elapsed_time_ms=0.0f;
                cudaEvent_t start, stop;
                cudaEventCreate( &start );
                cudaEventCreate( &stop  );
                cudaEventRecord( start, 0 );
                std::cout << "  ...deleting duplicate hits";
        #endif /* TIME_ATTACK */
	using namespace thrust;
	if(targetIDArray.size() > 1) {
		/* do remove */
		const int new_size = remove_if(
				make_zip_iterator(
						make_tuple(
								targetIDArray   .begin() + 1,
								targetIndexArray.begin() + 1,
								queryIDArray    .begin() + 1,
								queryIndexArray .begin() + 1,
								tHitLengthArray .begin() + 1,
								qHitLengthArray .begin() + 1,
								matchNumArray   .begin() + 1,
								scoreArray      .begin() + 1,
								evalueArray     .begin() + 1
						)
				),
				make_zip_iterator(
						make_tuple(
								targetIDArray   .end(),
								targetIndexArray.end(),
								queryIDArray    .end(),
								queryIndexArray .end(),
								tHitLengthArray .end(),
								qHitLengthArray .end(),
								matchNumArray   .end(),
								scoreArray      .end(),
								evalueArray     .end()
						)
				),
				make_zip_iterator(
						make_tuple(
								targetIDArray   .begin() + 1,
								targetIndexArray.begin() + 1,
								queryIDArray    .begin() + 1,
								queryIndexArray .begin() + 1,
								targetIDArray   .begin(),
								targetIndexArray.begin(),
								queryIDArray    .begin(),
								queryIndexArray .begin()
						)
				),
				is_duplicate()
		) - make_zip_iterator(
				make_tuple(
						targetIDArray   .begin(),
						targetIndexArray.begin(),
						queryIDArray    .begin(),
						queryIndexArray .begin(),
						tHitLengthArray .begin(),
						qHitLengthArray .begin(),
						matchNumArray   .begin(),
						scoreArray      .begin(),
						evalueArray     .begin()
				)
		);

		/* resize */
		targetIDArray   .resize(new_size);
		targetIndexArray.resize(new_size);
		queryIDArray    .resize(new_size);
		queryIndexArray .resize(new_size);
		tHitLengthArray .resize(new_size);
		qHitLengthArray .resize(new_size);
		matchNumArray   .resize(new_size);
		scoreArray      .resize(new_size);
		evalueArray     .resize(new_size);
	}
        #ifdef TIME_ATTACK
                std::cout << "...............................finished.";
                cudaEventRecord( stop, 0 );
                cudaEventSynchronize( stop );
                cudaEventElapsedTime( &elapsed_time_ms, start, stop );
                std::cout
                                << " (costs " << elapsed_time_ms << "ms) "
                                << targetIDArray.size() << " hits found."
                                << std::endl;
        #endif /* TIME_ATTACK */
}

/********************************* non class functions ***********************************/

void addResult(
		const thrust::device_vector<int>& add_targetIDArray,
		const thrust::device_vector<int>& add_targetIndexArray,
		const thrust::device_vector<int>& add_queryIDArray,
		const thrust::device_vector<int>& add_queryIndexArray,
		const thrust::device_vector<int>& add_tHitLengthArray,
		const thrust::device_vector<int>& add_qHitLengthArray,
		const thrust::device_vector<int>& add_matchNumArray,
		const thrust::device_vector<int>& add_scoreArray,
		const thrust::device_vector<double>& add_evalueArray,
		thrust::device_vector<int>& targetIDArray,
		thrust::device_vector<int>& targetIndexArray,
		thrust::device_vector<int>& queryIDArray,
		thrust::device_vector<int>& queryIndexArray,
		thrust::device_vector<int>& tHitLengthArray,
		thrust::device_vector<int>& qHitLengthArray,
		thrust::device_vector<int>& matchNumArray,
		thrust::device_vector<int>& scoreArray,
		thrust::device_vector<double>& evalueArray) {
        #ifdef TIME_ATTACK
                float elapsed_time_ms=0.0f;
                cudaEvent_t start, stop;
                cudaEventCreate( &start );
                cudaEventCreate( &stop  );
                cudaEventRecord( start, 0 );
                std::cout << "  ...writing un-exact matches";
        #endif /* TIME_ATTACK */
	const int oldResultSize = targetIDArray.size();
	const int newResultSize = oldResultSize + add_targetIDArray.size();
	targetIDArray   .resize(newResultSize);
	targetIndexArray.resize(newResultSize);
	queryIDArray    .resize(newResultSize);
	queryIndexArray .resize(newResultSize);
	tHitLengthArray .resize(newResultSize);
	qHitLengthArray .resize(newResultSize);
	matchNumArray   .resize(newResultSize);
	scoreArray      .resize(newResultSize);
	evalueArray     .resize(newResultSize);
	copy(
			add_targetIDArray.begin(),
			add_targetIDArray.end(),
			targetIDArray    .begin() + oldResultSize
	);
	copy(
			add_targetIndexArray.begin(),
			add_targetIndexArray.end(),
			targetIndexArray    .begin() + oldResultSize
	);
	copy(
			add_queryIDArray.begin(),
			add_queryIDArray.end(),
			queryIDArray    .begin() + oldResultSize
	);
	copy(
			add_queryIndexArray.begin(),
			add_queryIndexArray.end(),
			queryIndexArray    .begin() + oldResultSize
	);
	copy(
			add_tHitLengthArray.begin(),
			add_tHitLengthArray.end(),
			tHitLengthArray    .begin() + oldResultSize
	);
	copy(
			add_qHitLengthArray.begin(),
			add_qHitLengthArray.end(),
			qHitLengthArray    .begin() + oldResultSize
	);
	copy(
			add_matchNumArray.begin(),
			add_matchNumArray.end(),
			matchNumArray    .begin() + oldResultSize
	);
	copy(
			add_scoreArray.begin(),
			add_scoreArray.end(),
			scoreArray    .begin() + oldResultSize
	);
	copy(
			add_evalueArray.begin(),
			add_evalueArray.end(),
			evalueArray    .begin() + oldResultSize
	);
        #ifdef TIME_ATTACK
                std::cout << "..............................finished.";
                cudaEventRecord( stop, 0 );
                cudaEventSynchronize( stop );
                cudaEventElapsedTime( &elapsed_time_ms, start, stop );
                std::cout
                                << " (costs " << elapsed_time_ms << "ms) "
                                << add_targetIDArray.size() << " un-exact hits found."
                                << std::endl;
        #endif /* TIME_ATTACK */
}

/************************************* class functions **************************************/
CDeviceHitList::CDeviceHitList(
		const CHostSetting& s,
		const CDeviceHashTable& h,
		const CDeviceSeqList_query& q,
		const int t_begin,
		const int q_begin) {
	using namespace thrust;

	/* prepare seed hit list */
	device_vector<int> seed_targetIDArray;
	device_vector<int> seed_targetIndexArray;
	device_vector<int> seed_queryIDArray;
	device_vector<int> seed_queryIndexArray;

	createRawSeedList(
			s,
			h,
			q,
			t_begin,
			q_begin,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
	deleteSeedsOnSequenceBoundary(
			s,
			h,
			q,
			t_begin,
			q_begin,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);

	sortSeeds(
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
	deletingIsolateSeeds(
			s.getAllowableWidth(),
			s.getAllowableGap(),
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
	deleteDuplicateSeeds(
			s,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);

	/* prepare hit list array */
	device_vector<int> seed_tLengthArray (seed_targetIDArray.size(), s.getLMerLength());
	device_vector<int> seed_qLengthArray (seed_targetIDArray.size(), s.getLMerLength());
	device_vector<int> seed_matchNumArray(seed_targetIDArray.size(), s.getLMerLength());
	device_vector<int> seed_scoreArray   (seed_targetIDArray.size(), s.getLMerLength() * MATCH_POINT);

	alignmentHits(
			s,
			h,
			q,
			t_begin,
			q_begin,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray,
			seed_tLengthArray,
			seed_qLengthArray,
			seed_matchNumArray,
			seed_scoreArray);

	/* prepare calculation of E-value */
	thrust::device_vector<double> seed_evalueArray(seed_targetIDArray.size());

	#ifdef TIME_ATTACK
		float elapsed_time_ms=0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate( &start );
		cudaEventCreate( &stop  );
		cudaEventRecord( start, 0 );
		std::cout << "  ...calculating E-value";
	#endif /* TIME_ATTACK */
	const int blockDim_x = 256;
	const int seedAlignmentNum = seed_targetIDArray.size();
	calculateEvalue<<<(seedAlignmentNum/blockDim_x)+1, blockDim_x>>>(
			q_begin,
			seedAlignmentNum,
			s.getTotalDatabaseSize(),
			s.getK(),
			s.getLambda(),
			raw_pointer_cast( &*q.getLengthArray().begin() ),
			raw_pointer_cast( &*seed_queryIDArray .begin() ),
			raw_pointer_cast( &*seed_scoreArray   .begin() ),
			raw_pointer_cast( &*seed_evalueArray  .begin() )
	);
	#ifdef TIME_ATTACK
		std::cout << "...................................finished.";
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		std::cout
				<< " (costs " << elapsed_time_ms << "ms) "
				<< seed_targetIDArray.size() << " hits found."
				<< std::endl;
	#endif /* TIME ATTACK */

	if(s.getCutOff() != -1) {
		deleteHits_lowEValue(
				s.getCutOff(),
				seed_targetIDArray,
				seed_targetIndexArray,
				seed_queryIDArray,
				seed_queryIndexArray,
				seed_tLengthArray,
				seed_qLengthArray,
				seed_matchNumArray,
				seed_scoreArray,
				seed_evalueArray);
	}

	deleteHits_tooShort(
			s,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray,
			seed_tLengthArray,
			seed_qLengthArray,
			seed_matchNumArray,
			seed_scoreArray,
			seed_evalueArray);

	if(seed_targetIDArray.size() != 0) {
		addResult(
				seed_targetIDArray,
				seed_targetIndexArray,
				seed_queryIDArray,
				seed_queryIndexArray,
				seed_tLengthArray,
				seed_qLengthArray,
				seed_matchNumArray,
				seed_scoreArray,
				seed_evalueArray,
				targetIDArray,
				targetIndexArray,
				queryIDArray,
				queryIndexArray,
				tHitLengthArray,
				qHitLengthArray,
				matchNumArray,
				scoreArray,
				evalueArray);
	}

	#ifdef TIME_ATTACK
		cudaEventCreate( &start );
		cudaEventCreate( &stop  );
		cudaEventRecord( start, 0 );
		std::cout << "  ...sorting results";
	#endif /* TIME_ATTACK */
	resultSorting(
			targetIDArray,
			targetIndexArray,
			queryIDArray,
			queryIndexArray,
			tHitLengthArray,
			qHitLengthArray,
			matchNumArray,
			scoreArray,
			evalueArray);
	#ifdef TIME_ATTACK
		std::cout << ".......................................finished.";
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		std::cout
				<< " (costs " << elapsed_time_ms << "ms) "
				<< targetIDArray.size() << " total hits found."
				<< std::endl;
	#endif /* TIME_ATTACK */

	deleteHits_duplicateResult(
			targetIDArray,
			targetIndexArray,
			queryIDArray,
			queryIndexArray,
			tHitLengthArray,
			qHitLengthArray,
			matchNumArray,
			scoreArray,
			evalueArray);
}

void CDeviceHitList::getResult(CHostResultHolder& holder) {
	#ifdef TIME_ATTACK
		float elapsed_time_ms=0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate( &start );
		cudaEventCreate( &stop  );
		cudaEventRecord( start, 0 );
		std::cout << "  ...Record result to host";
	#endif /* TIME_ATTACK */
	holder.addResult(
			targetIDArray,
			targetIndexArray,
			queryIDArray,
			queryIndexArray,
			tHitLengthArray,
			qHitLengthArray,
			matchNumArray,
			scoreArray,
			evalueArray);
	#ifdef TIME_ATTACK
		std::cout << ".................................finished.";
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		std::cout << " (costs " << elapsed_time_ms << "ms)" << std::endl;
	#endif /* TIME_ATTACK */
}
