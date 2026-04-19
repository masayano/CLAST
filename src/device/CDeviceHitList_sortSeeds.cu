#include "device/CDeviceHitList_sortSeeds.cuh"
#include "device/CDeviceHitList_seed_host_api.cuh"

#include "util/common.hpp"

#ifdef TIME_ATTACK
	#include <iostream>
#endif

#ifdef MODE_TEST
	#include "test_support/CTest.cuh"
#endif

void sortSeeds(
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
		std::cout << "  ...measure distance sorting";
	#endif /* TIME_ATTACK */

	measureDistanceSorting(
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);

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
