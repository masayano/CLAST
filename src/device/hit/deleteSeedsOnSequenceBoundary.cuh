#ifndef C_DEVICE_HIT_LIST_DELETE_SEEDS_ON_SEQUENCE_BOUNDARY_CUH_
#define C_DEVICE_HIT_LIST_DELETE_SEEDS_ON_SEQUENCE_BOUNDARY_CUH_

#include "host/CHostSetting.cuh"
#include "device/CDeviceHashTable.cuh"
#include "device/seq/query.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

void deleteSeedsOnSequenceBoundary(
		const CHostSetting& s,
		const CDeviceHashTable& h,
		const CDeviceSeqList_query& q,
		const int t_begin,
		const int q_begin,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray);

namespace clast::hit {

struct is_onBoundary {
	template <class Tuple>
	__host__ __device__ bool operator() (const Tuple& tuple) const {
		const int kMerLength = thrust::get<0>(tuple);
		const int length     = thrust::get<1>(tuple);
		const int idx        = thrust::get<2>(tuple);
		return (kMerLength + idx > length);
	}
};

struct is_onCorner {
	template <class Tuple>
	__host__ __device__ bool operator() (const Tuple& tuple) const {
		using namespace thrust;
		const int allowableGap = get<0>(tuple);
		const int tLength      = get<1>(tuple);
		const int tIdx         = get<2>(tuple);
		const int qLength      = get<3>(tuple);
		const int qIdx         = get<4>(tuple);
		const int tHitStartIdx = tIdx - qIdx;
		if(tHitStartIdx + allowableGap < 0) {
			return true;
		} else if(tHitStartIdx + qLength - allowableGap > tLength) {
			return true;
		} else {
			return false;
		}
	}
};

template <class ThrustVector>
void onQueryBoundary(
		const int kMerLength,
		const int q_begin,
		const ThrustVector& qLengthArray,
		ThrustVector& seed_targetIDArray,
		ThrustVector& seed_targetIndexArray,
		ThrustVector& seed_queryIDArray,
		ThrustVector& seed_queryIndexArray) {
	using namespace thrust;

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

	seed_targetIDArray   .resize(new_size);
	seed_targetIndexArray.resize(new_size);
	seed_queryIDArray    .resize(new_size);
	seed_queryIndexArray .resize(new_size);
}

template <class ThrustVector>
void onTargetBoundary(
		const int kMerLength,
		const int t_begin,
		const ThrustVector& tLengthArray,
		ThrustVector& seed_targetIDArray,
		ThrustVector& seed_targetIndexArray,
		ThrustVector& seed_queryIDArray,
		ThrustVector& seed_queryIndexArray) {
	using namespace thrust;

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

	seed_targetIDArray   .resize(new_size);
	seed_targetIndexArray.resize(new_size);
	seed_queryIDArray    .resize(new_size);
	seed_queryIndexArray .resize(new_size);
}

template <class ThrustVector>
void onCorner(
		const int allowableGap,
		const int t_begin,
		const int q_begin,
		const ThrustVector& tLengthArray,
		const ThrustVector& qLengthArray,
		ThrustVector& seed_targetIDArray,
		ThrustVector& seed_targetIndexArray,
		ThrustVector& seed_queryIDArray,
		ThrustVector& seed_queryIndexArray) {
	using namespace thrust;

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
	seed_targetIDArray   .resize(new_size);
	seed_targetIndexArray.resize(new_size);
	seed_queryIDArray    .resize(new_size);
	seed_queryIndexArray .resize(new_size);
}

} // namespace clast::hit

#endif
