#include "device/hit/deleteDuplicateSeeds.cuh"
#include "device/hit/deleteIsolateSeeds.cuh"
#include "device/hit/deleteSeedsOnSequenceBoundary.cuh"

#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
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

void deleteSeedHasNotNearPair(
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

void onQueryBoundary(
		const int kMerLength,
		const int q_begin,
		const thrust::host_vector<int> qLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
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

void onTargetBoundary(
		const int kMerLength,
		const int t_begin,
		const thrust::host_vector<int> tLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
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

void onCorner(
		const int allowableGap,
		const int t_begin,
		const int q_begin,
		const thrust::host_vector<int> tLengthArray,
		const thrust::host_vector<int> qLengthArray,
		thrust::host_vector<int>& seed_targetIDArray,
		thrust::host_vector<int>& seed_targetIndexArray,
		thrust::host_vector<int>& seed_queryIDArray,
		thrust::host_vector<int>& seed_queryIndexArray) {
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
