#ifndef C_DEVICE_HIT_LIST_DELETE_ISOLATE_SEEDS_CUH_
#define C_DEVICE_HIT_LIST_DELETE_ISOLATE_SEEDS_CUH_

#include <thrust/device_vector.h>

void deletingIsolateSeeds(
		const int allowableWidth,
		const int allowableGap,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray);

struct hasNotNearPair {
	template <typename Tuple>
	__host__ bool operator() (const Tuple& tuple) const {
		using namespace thrust;
		const int tID  = get<0>(tuple);
		const int tIdx = get<1>(tuple);
		const int qID  = get<2>(tuple);
		const int qIdx = get<3>(tuple);
		const int post_tID  = get<4>(tuple);
		const int post_tIdx = get<5>(tuple);
		const int post_qID  = get<6>(tuple);
		const int post_qIdx = get<7>(tuple);
		const int allowableWidth = get<8>(tuple);
		const int allowableGap   = get<9>(tuple);
		if((qID == post_qID) && (tID == post_tID)) {
			if((allowableWidth >= (tIdx - post_tIdx)) && (allowableWidth >= (post_tIdx - tIdx))) {
				const int post_diagonalIdx = post_tIdx - post_qIdx;
				const int diagonalIdx      = tIdx - qIdx;
				if(allowableGap >= (post_diagonalIdx - diagonalIdx)) {
					return false;
				}
			}
		}
		return true;
	}
};

#endif
