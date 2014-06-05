#include "utilResultSorting.cuh"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

void resultSorting(
		thrust::device_vector<int>& targetIDArray,
		thrust::device_vector<int>& targetIndexArray,
		thrust::device_vector<int>& queryIDArray,
		thrust::device_vector<int>& queryIndexArray,
		thrust::device_vector<int>& tHitLengthArray,
		thrust::device_vector<int>& qHitLengthArray,
		thrust::device_vector<int>& matchNumArray,
		thrust::device_vector<int>& scoreArray,
		thrust::device_vector<double>& evalueArray) {
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
