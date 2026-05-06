#ifndef C_DEVICE_HIT_LIST_SORT_SEEDS_CUH_
#define C_DEVICE_HIT_LIST_SORT_SEEDS_CUH_

#include <thrust/device_vector.h>

void sortSeeds(
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray);

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

namespace clast::hit {

struct measure_distance {
	template <class Tuple1, class Tuple2>
	__host__ __device__ bool operator() (const Tuple1& tuple1, const Tuple2& tuple2) const {
		using namespace thrust;

		const int tID_1  = get<0>(tuple1);
		const int tIdx_1 = get<1>(tuple1);
		const int qID_1  = get<2>(tuple1);
		const int qIdx_1 = get<3>(tuple1);

		const int tID_2  = get<0>(tuple2);
		const int tIdx_2 = get<1>(tuple2);
		const int qID_2  = get<2>(tuple2);
		const int qIdx_2 = get<3>(tuple2);

		const int dia_1 = tIdx_1 - qIdx_1;
		const int dia_2 = tIdx_2 - qIdx_2;

		if(qID_1 < qID_2) {
			return true;
		} else if(qID_1 == qID_2) {
			if(tID_1 < tID_2) {
				return true;
			} else if(tID_1 == tID_2) {
				if(dia_1 < dia_2) {
					return true;
				} else if(dia_1 == dia_2) {
					if(qIdx_1 < qIdx_2) {
						return true;
					} else {
						return false;
					}
				} else {
					return false;
				}
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
};

template <class ThrustVector>
void measureDistanceSorting(
		ThrustVector& seed_targetIDArray,
		ThrustVector& seed_targetIndexArray,
		ThrustVector& seed_queryIDArray,
		ThrustVector& seed_queryIndexArray) {
	using namespace thrust;

	thrust::sort(
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
			measure_distance()
	);
}

} // namespace clast::hit

#endif
