#ifndef UTIL_RESULT_SORTING_CUH_
#define UTIL_RESULT_SORTING_CUH_

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace clast::sorting {

struct result {
	template <class Tuple1, class Tuple2>
	__host__ __device__ bool operator() (const Tuple1& tuple1, const Tuple2& tuple2) const {
		using namespace thrust;
		const int qID_1  = get<2>(tuple1) / 2; // magic number "2" ... "+" query and "-" query
		const int scr_1  = get<7>(tuple1);
		const int qID_2  = get<2>(tuple2) / 2; // ditto
		const int scr_2  = get<7>(tuple2);
		if(qID_1 < qID_2) {
			return true;
		} else if(qID_1 == qID_2) {
			if(scr_1 > scr_2) {
				return true;
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
};

template <class ThrustVectorInt, class ThrustVectorDouble>
void resultSorting(
		ThrustVectorInt& targetIDArray,
		ThrustVectorInt& targetIndexArray,
		ThrustVectorInt& queryIDArray,
		ThrustVectorInt& queryIndexArray,
		ThrustVectorInt& tHitLengthArray,
		ThrustVectorInt& qHitLengthArray,
		ThrustVectorInt& matchNumArray,
		ThrustVectorInt& scoreArray,
		ThrustVectorDouble& evalueArray) {
	using namespace thrust;
	thrust::sort(
			make_zip_iterator(
					make_tuple(
							targetIDArray   .begin(),
							targetIndexArray.begin(),
							queryIDArray    .begin(),
							queryIndexArray .begin(),
							tHitLengthArray .begin(),
							qHitLengthArray .begin(),
							matchNumArray   .begin(),
							scoreArray      .begin(),
							evalueArray     .begin()
					)
			),
			make_zip_iterator(
					make_tuple(
							targetIDArray   .end(),
							targetIndexArray.end(),
							queryIDArray    .end(),
							queryIndexArray .end(),
							tHitLengthArray .end(),
							qHitLengthArray .end(),
							matchNumArray   .end(),
							scoreArray      .end(),
							evalueArray     .end()
					)
			),
			result()
	);
}

} // namespace clast::sorting

#endif
