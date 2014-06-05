#ifndef UTIL_RESULT_SORTING_CUH_
#define UTIL_RESULT_SORTING_CUH_

#include <thrust/device_vector.h>

typedef thrust::tuple<int,int,int,int,int,int,int,int,double> Hit;
struct result : public thrust::binary_function<Hit,Hit,bool> {
	__device__ bool operator() (const Hit& tuple1, const Hit& tuple2) const {
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

void resultSorting(
		thrust::device_vector<int>& targetIDArray,
		thrust::device_vector<int>& targetIndexArray,
		thrust::device_vector<int>& queryIDArray,
		thrust::device_vector<int>& queryIndexArray,
		thrust::device_vector<int>& tHitLengthArray,
		thrust::device_vector<int>& qHitLengthArray,
		thrust::device_vector<int>& matchNumArray,
		thrust::device_vector<int>& scoreArray,
		thrust::device_vector<double>& evalueArray);

#endif
