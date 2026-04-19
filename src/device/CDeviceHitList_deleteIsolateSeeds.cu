#include "device/CDeviceHitList_deleteIsolateSeeds.cuh"
#include "device/CDeviceHitList_seed_host_api.cuh"

#include "util/common.hpp"

#ifdef TIME_ATTACK
	#include <iostream>
#endif

#ifdef MODE_TEST
	#include "test_support/CTest.cuh"
#endif

#include <thrust/host_vector.h>

#include <thrust/functional.h>

void deletingIsolateSeeds(
		const int allowableWidth,
		const int allowableGap,
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
		std::cout << "  ...deleting isolate seeds";
	#endif /* TIME_ATTACK */
	using namespace thrust;
	host_vector<int> h_tIDArray  = seed_targetIDArray;
	host_vector<int> h_tIdxArray = seed_targetIndexArray;
	host_vector<int> h_qIDArray  = seed_queryIDArray;
	host_vector<int> h_qIdxArray = seed_queryIndexArray;
	deleteSeedHasNotNearPair(
			allowableWidth,
			allowableGap,
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
		std::cout << "................................finished.";
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
