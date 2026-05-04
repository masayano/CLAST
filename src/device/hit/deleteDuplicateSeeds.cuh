#ifndef C_DEVICE_HIT_LIST_DELETE_DUPLICATE_SEEDS_CUH_
#define C_DEVICE_HIT_LIST_DELETE_DUPLICATE_SEEDS_CUH_

#include "host/CHostSetting.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

void deleteDuplicateSeeds(
		const CHostSetting& s,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray);

namespace clast::hit {

struct hasNearPair {
	template <class Tuple>
	__host__ __device__ bool operator() (const Tuple& tuple) const {
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
					return true;
				}
			}
		}
		return false;
	}
};

template <class ThrustVector>
void deleteDuplicateSeeds(
		const int allowableWidth,
		const int allowableGap,
		ThrustVector& seed_targetIDArray,
		ThrustVector& seed_targetIndexArray,
		ThrustVector& seed_queryIDArray,
		ThrustVector& seed_queryIndexArray) {
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

} // namespace clast::hit

#endif
