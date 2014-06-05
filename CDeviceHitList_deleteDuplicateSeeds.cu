#include "CDeviceHitList_deleteDuplicateSeeds.cuh"

#include "common.hpp"
#ifdef TIME_ATTACK
	#include <iostream>
#endif
#ifdef MODE_TEST
	#include "CTest.cuh"
#endif

/*********************************** private *****************************************/
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>

void deleteDuplicateSeeds(
		const int allowableWidth,
		const int allowableGap,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
	using namespace thrust;

	if(seed_targetIDArray.size() > 1) {
		const int newSize = remove_if(
				make_zip_iterator(
						make_tuple(
								seed_targetIDArray   .begin() + 1,
								seed_targetIndexArray.begin() + 1,
								seed_queryIDArray    .begin() + 1,
								seed_queryIndexArray .begin() + 1
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
								seed_targetIDArray   .begin() + 1,
								seed_targetIndexArray.begin() + 1,
								seed_queryIDArray    .begin() + 1,
								seed_queryIndexArray .begin() + 1,
								seed_targetIDArray   .begin(),
								seed_targetIndexArray.begin(),
								seed_queryIDArray    .begin(),
								seed_queryIndexArray .begin(),
								make_constant_iterator(allowableWidth),
								make_constant_iterator(allowableGap)
						)
				),
				hasNearPair()
		) - make_zip_iterator(
				make_tuple(
						seed_targetIDArray   .begin(),
						seed_targetIndexArray.begin(),
						seed_queryIDArray    .begin(),
						seed_queryIndexArray .begin()
				)
		);

		seed_targetIDArray   .resize(newSize);
		seed_targetIndexArray.resize(newSize);
		seed_queryIDArray    .resize(newSize);
		seed_queryIndexArray .resize(newSize);
	}
}

/*********************************** public *****************************************/

void deleteDuplicateSeeds(
		const CHostSetting& s,
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
		std::cout << "  ...deleting duplicate seeds";
	#endif /* TIME_ATTACK */
	using namespace thrust;
	host_vector<int> h_tIDArray  = seed_targetIDArray;
	host_vector<int> h_tIdxArray = seed_targetIndexArray;
	host_vector<int> h_qIDArray  = seed_queryIDArray;
	host_vector<int> h_qIdxArray = seed_queryIndexArray;
	deleteDuplicateSeeds(
			s.getAllowableWidth(),
			s.getAllowableGap(),
			h_tIDArray,
			h_tIdxArray,
			h_qIDArray,
			h_qIdxArray);
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
		std::cout << "..............................finished.";
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
