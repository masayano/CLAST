#include "CDeviceHitList_deleteSeedsOnSequenceBoundary.cuh"

#include "common.hpp"

#ifdef TIME_ATTACK
	#include <iostream>
#endif

#ifdef MODE_TEST
	#include "CTest.cuh"
#endif

/*************************************** private *****************************************/

#include <thrust/remove.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
void onQueryBoundary(
		const int kMerLength,
		const int q_begin,
		const thrust::host_vector<int> qLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
	using namespace thrust;

	/* do remove */
	const int new_size = remove_if(
			make_zip_iterator(
					make_tuple(
							seed_targetIDArray   .begin(),
							seed_targetIndexArray.begin(),
							seed_queryIDArray    .begin(),
							seed_queryIndexArray .begin()
					)
			),
			make_zip_iterator(
					make_tuple(
							seed_targetIDArray   .end(),
							seed_targetIndexArray.end(),
							seed_queryIDArray    .end(),
							seed_queryIndexArray .end()
					)
			),
			make_zip_iterator(
					make_tuple(
							make_constant_iterator(kMerLength),
							make_permutation_iterator(
									qLengthArray     .begin() - q_begin,
									seed_queryIDArray.begin()
							),
							seed_queryIndexArray.begin()
					)
			),
			is_onBoundary()
	) - make_zip_iterator(
			make_tuple(
					seed_targetIDArray   .begin(),
					seed_targetIndexArray.begin(),
					seed_queryIDArray    .begin(),
					seed_queryIndexArray .begin()
			)
	);

	/* resize */
	seed_targetIDArray   .resize(new_size);
	seed_targetIndexArray.resize(new_size);
	seed_queryIDArray    .resize(new_size);
	seed_queryIndexArray .resize(new_size);
}

void onTargetBoundary(
		const int kMerLength,
		const int t_begin,
		const thrust::host_vector<int> tLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
	using namespace thrust;

	/* do remove */
	const int new_size = remove_if(
			make_zip_iterator(
					make_tuple(
							seed_targetIDArray   .begin(),
							seed_targetIndexArray.begin(),
							seed_queryIDArray    .begin(),
							seed_queryIndexArray .begin()
					)
			),
			make_zip_iterator(
					make_tuple(
							seed_targetIDArray   .end(),
							seed_targetIndexArray.end(),
							seed_queryIDArray    .end(),
							seed_queryIndexArray .end()
					)
			),
			make_zip_iterator(
					make_tuple(
							make_constant_iterator(kMerLength),
							make_permutation_iterator(
									tLengthArray      .begin() - t_begin,
									seed_targetIDArray.begin()
							),
							seed_targetIndexArray.begin()
					)
			),
			is_onBoundary()
	) - make_zip_iterator(
			make_tuple(
					seed_targetIDArray   .begin(),
					seed_targetIndexArray.begin(),
					seed_queryIDArray    .begin(),
					seed_queryIndexArray .begin()
			)
	);

	/* resize */
	seed_targetIDArray   .resize(new_size);
	seed_targetIndexArray.resize(new_size);
	seed_queryIDArray    .resize(new_size);
	seed_queryIndexArray .resize(new_size);
}

void onCorner(
		const int allowableGap,
		const int t_begin,
		const int q_begin,
		const thrust::host_vector<int> tLengthArray,
		const thrust::host_vector<int> qLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
	using namespace thrust;

	/* do remove */
	const int new_size = remove_if(
			make_zip_iterator(
					make_tuple(
							seed_targetIDArray   .begin(),
							seed_targetIndexArray.begin(),
							seed_queryIDArray    .begin(),
							seed_queryIndexArray .begin()
					)
			),
			make_zip_iterator(
					make_tuple(
							seed_targetIDArray   .end(),
							seed_targetIndexArray.end(),
							seed_queryIDArray    .end(),
							seed_queryIndexArray .end()
					)
			),
			make_zip_iterator(
					make_tuple(
							make_constant_iterator(allowableGap),
							make_permutation_iterator(
									tLengthArray      .begin() - t_begin,
									seed_targetIDArray.begin()
							),
							seed_targetIndexArray.begin(),
							make_permutation_iterator(
									qLengthArray     .begin() - q_begin,
									seed_queryIDArray.begin()
							),
							seed_queryIndexArray.begin()
					)
			),
			is_onCorner()
	) - make_zip_iterator(
			make_tuple(
					seed_targetIDArray   .begin(),
					seed_targetIndexArray.begin(),
					seed_queryIDArray    .begin(),
					seed_queryIndexArray .begin()
			)
	);
	/* resize */
	seed_targetIDArray   .resize(new_size);
	seed_targetIndexArray.resize(new_size);
	seed_queryIDArray    .resize(new_size);
	seed_queryIndexArray .resize(new_size);
}

/*************************************** public *****************************************/

void deleteSeedsOnSequenceBoundary(
		const CHostSetting& s,
		const CDeviceHashTable& h,
		const CDeviceSeqList_query& q,
		const int t_begin,
		const int q_begin,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray) {
	#ifdef TIME_ATTACK
		float elapsed_time_ms=0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate( &start );
		cudaEventCreate( &stop  );
		cudaEventRecord( start, 0 );
		std::cout << "  ...deleting hits on sequence boundary";
	#endif /* TIME_ATTACK */
	using namespace thrust;
	host_vector<int> h_tIDArray  = seed_targetIDArray;
	host_vector<int> h_tIdxArray = seed_targetIndexArray;
	host_vector<int> h_qIDArray  = seed_queryIDArray;
	host_vector<int> h_qIdxArray = seed_queryIndexArray;
	onQueryBoundary(
			s.getLMerLength(),
			q_begin,
			q.getLengthArray(),
			h_tIDArray,
			h_tIdxArray,
			h_qIDArray,
			h_qIdxArray);
	onTargetBoundary(
			s.getLMerLength(),
			t_begin,
			h.getTarget().getLengthArray(),
			h_tIDArray,
			h_tIdxArray,
			h_qIDArray,
			h_qIdxArray);
	if(!s.getFlgLocal()) {
		onCorner(
			s.getAllowableGap(),
			t_begin,
			q_begin,
			h.getTarget().getLengthArray(),
			q.getLengthArray(),
			h_tIDArray,
			h_tIdxArray,
			h_qIDArray,
			h_qIdxArray);
	}
	seed_targetIDArray    = h_tIDArray;
	seed_targetIndexArray = h_tIdxArray;
	seed_queryIDArray     = h_qIDArray;
	seed_queryIndexArray  = h_qIdxArray;
	const int newSize = h_tIDArray.size();
	seed_targetIDArray   .resize(newSize);
	seed_targetIndexArray.resize(newSize);
	seed_queryIDArray    .resize(newSize);
	seed_queryIndexArray .resize(newSize);
	#ifdef TIME_ATTACK
		std::cout << "....................finished.";
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		std::cout
				<< " (costs " << elapsed_time_ms << "ms) "
				<< seed_targetIDArray.size() << " hits found."
				<< std::endl;
	#endif /* TIME_ATTACK */
	#ifdef MODE_TEST
		CTest::printQueryToTarget(
				seed_targetIDArray,
				seed_targetIndexArray,
				seed_queryIDArray,
				seed_queryIndexArray);
	#endif /* MODE_TEST */
}
