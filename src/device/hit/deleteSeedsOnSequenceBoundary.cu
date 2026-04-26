#include "device/hit/deleteSeedsOnSequenceBoundary.cuh"
#include "device/hit/seedHostApi.cuh"

#include "util/common.hpp"
#include "util/time_attack.hpp"
#include <iostream>

#ifdef MODE_TEST
	#include "test_support/CTest.cuh"
#endif

#include <thrust/host_vector.h>

void removeSeedsOnSequenceBoundary(
		const CHostSetting& s,
		const CDeviceHashTable& h,
		const CDeviceSeqList_query& q,
		const int t_begin,
		const int q_begin,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray) {
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
	if (!s.getFlgLocal()) {
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
}

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
	time_attack::runLabeledWithSuffix(
			"  ...deleting hits on sequence boundary", "....................finished.",
			[&] {
				removeSeedsOnSequenceBoundary(
						s,
						h,
						q,
						t_begin,
						q_begin,
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
