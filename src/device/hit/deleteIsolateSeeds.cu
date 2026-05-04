#include "device/hit/deleteIsolateSeeds.cuh"

#include "util/common.hpp"
#include "util/time_attack.hpp"
#include <iostream>

#ifdef MODE_TEST
	#include "test_support/CTest.cuh"
#endif

#include <thrust/host_vector.h>

void deletingIsolateSeedsImpl(
		const int allowableWidth,
		const int allowableGap,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray) {
	using namespace thrust;
	host_vector<int> h_tIDArray  = seed_targetIDArray;
	host_vector<int> h_tIdxArray = seed_targetIndexArray;
	host_vector<int> h_qIDArray  = seed_queryIDArray;
	host_vector<int> h_qIdxArray = seed_queryIndexArray;
	clast::hit::deleteSeedHasNotNearPair(
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
}

void deletingIsolateSeeds(
		const int allowableWidth,
		const int allowableGap,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray) {
	time_attack::runLabeledWithSuffix(
			"  ...deleting isolate seeds", "................................finished.",
			[&] {
				deletingIsolateSeedsImpl(
						allowableWidth,
						allowableGap,
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
	#endif /* MODE_TEST */
}
