#ifndef C_DEVICE_HIT_LIST_DELETE_BAD_HITS_CUH_
#define C_DEVICE_HIT_LIST_DELETE_BAD_HITS_CUH_

#include "CHostSetting.cuh"

#include <thrust/device_vector.h>

void deleteHits_lowEValue(
		const double cutOffEValue,
		thrust::device_vector<int>& targetIDArray,
		thrust::device_vector<int>& targetIndexArray,
		thrust::device_vector<int>& queryIDArray,
		thrust::device_vector<int>& queryIndexArray,
		thrust::device_vector<int>& tHitLengthArray,
		thrust::device_vector<int>& qHitLengthArray,
		thrust::device_vector<int>& matchNumArray,
		thrust::device_vector<int>& scoreArray,
		thrust::device_vector<double>& evalueArray);

void deleteHits_tooShort(
		const CHostSetting& s,
		thrust::device_vector<int>& targetIDArray,
		thrust::device_vector<int>& targetIndexArray,
		thrust::device_vector<int>& queryIDArray,
		thrust::device_vector<int>& queryIndexArray,
		thrust::device_vector<int>& tHitLengthArray,
		thrust::device_vector<int>& qHitLengthArray,
		thrust::device_vector<int>& matchNumArray,
		thrust::device_vector<int>& scoreArray,
		thrust::device_vector<double>& evalueArray);

struct is_lowEValue {
	template <class Tuple>
	__host__ bool operator() (const Tuple& tuple) const {
		const double cutOffEValue = thrust::get<0>(tuple);
		const double eValue       = thrust::get<1>(tuple);
		return (cutOffEValue < eValue);
	}
};

struct is_tooShort {
	template <class Tuple>
	__host__ bool operator() (const Tuple& tuple) const {
		const int cutOffLength = thrust::get<0>(tuple);
		const int hitLength    = thrust::get<1>(tuple);
		return (cutOffLength > hitLength);
	}
};

#endif
