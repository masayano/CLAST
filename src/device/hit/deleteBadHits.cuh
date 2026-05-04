#ifndef C_DEVICE_HIT_LIST_DELETE_BAD_HITS_CUH_
#define C_DEVICE_HIT_LIST_DELETE_BAD_HITS_CUH_

#include "host/CHostSetting.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

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

namespace clast::hit {

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

template <class ThrustVectorInt, class ThrustVectorDouble>
void removeLowEValue(
		const double cutOffEValue,
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
	const int new_size = remove_if(
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
			make_zip_iterator(
					make_tuple(
							make_constant_iterator(cutOffEValue),
							evalueArray.begin()
					)
			),
			is_lowEValue()
	) - make_zip_iterator(
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
	);
	targetIDArray   .resize(new_size);
	targetIndexArray.resize(new_size);
	queryIDArray    .resize(new_size);
	queryIndexArray .resize(new_size);
	tHitLengthArray .resize(new_size);
	qHitLengthArray .resize(new_size);
	matchNumArray   .resize(new_size);
	scoreArray      .resize(new_size);
	evalueArray     .resize(new_size);
}

template <class ThrustVectorInt, class ThrustVectorDouble>
void removeTooShort(
		const int cutOffLength,
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
	const int new_size = remove_if(
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
			make_zip_iterator(
					make_tuple(
							make_constant_iterator(cutOffLength),
							qHitLengthArray.begin()
					)
			),
			is_tooShort()
	) - make_zip_iterator(
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
	);
	targetIDArray   .resize(new_size);
	targetIndexArray.resize(new_size);
	queryIDArray    .resize(new_size);
	queryIndexArray .resize(new_size);
	tHitLengthArray .resize(new_size);
	qHitLengthArray .resize(new_size);
	matchNumArray   .resize(new_size);
	scoreArray      .resize(new_size);
	evalueArray     .resize(new_size);
}

} // namespace clast::hit

#endif
