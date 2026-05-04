#ifndef CLAST_DEVICE_HIT_SEED_THRUST_VECTOR_OPS_CUH_
#define CLAST_DEVICE_HIT_SEED_THRUST_VECTOR_OPS_CUH_

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

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

struct hasNotNearPair {
	template <typename Tuple>
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
					return false;
				}
			}
		}
		return true;
	}
};

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

template <class ThrustVector>
void deleteSeedHasNotNearPair(
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
				hasNotNearPair()
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
									qLengthArray       .begin() - q_begin,
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
									tLengthArray       .begin() - t_begin,
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
									tLengthArray       .begin() - t_begin,
									seed_targetIDArray.begin()
							),
							seed_targetIndexArray.begin(),
							make_permutation_iterator(
									qLengthArray       .begin() - q_begin,
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
