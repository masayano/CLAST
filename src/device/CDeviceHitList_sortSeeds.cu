#include "device/CDeviceHitList_sortSeeds.cuh"
#include "device/CDeviceHitList_seed_host_api.cuh"

#include "util/common.hpp"
#include "util/time_attack.hpp"

#include <iostream>

#ifdef MODE_TEST
	#include "test_support/CTest.cuh"
#endif

void sortSeeds(
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray) {
	time_attack::runLabeledWithSuffix(
			"  ...measure distance sorting",
			"..............................finished.",
			[&] {
				measureDistanceSorting(
						seed_targetIDArray,
						seed_targetIndexArray,
						seed_queryIDArray,
						seed_queryIndexArray);
			},
			[&](std::ostream& os, float /*ms*/) {
				os << seed_targetIDArray.size() << " hits found." << std::endl;
			});
#ifdef MODE_TEST
	CTest::printQueryToTarget(
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
#endif
}
