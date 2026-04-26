#include "device/hit/deleteBadHits.cuh"

#include "util/common.hpp"
#include "util/time_attack.hpp"

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

void removeLowEValueHits(
		const double cutOffEValue,
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
	host_vector<int>    h_targetIDArray    = targetIDArray;
	host_vector<int>    h_targetIndexArray = targetIndexArray;
	host_vector<int>    h_queryIDArray     = queryIDArray;
	host_vector<int>    h_queryIndexArray  = queryIndexArray;
	host_vector<int>    h_tHitLengthArray  = tHitLengthArray;
	host_vector<int>    h_qHitLengthArray  = qHitLengthArray;
	host_vector<int>    h_matchNumArray    = matchNumArray;
	host_vector<int>    h_scoreArray       = scoreArray;
	host_vector<double> h_evalueArray      = evalueArray;
	/* remove */
	const int new_size = remove_if(
			make_zip_iterator(
					make_tuple(
							h_targetIDArray   .begin(),
							h_targetIndexArray.begin(),
							h_queryIDArray    .begin(),
							h_queryIndexArray .begin(),
							h_tHitLengthArray .begin(),
							h_qHitLengthArray .begin(),
							h_matchNumArray   .begin(),
							h_scoreArray      .begin(),
							h_evalueArray     .begin()
					)
			),
			make_zip_iterator(
					make_tuple(
							h_targetIDArray   .end(),
							h_targetIndexArray.end(),
							h_queryIDArray    .end(),
							h_queryIndexArray .end(),
							h_tHitLengthArray .end(),
							h_qHitLengthArray .end(),
							h_matchNumArray   .end(),
							h_scoreArray      .end(),
							h_evalueArray     .end()
					)
			),
			make_zip_iterator(
					make_tuple(
							make_constant_iterator(cutOffEValue),
							h_evalueArray.begin()
					)
			),
			is_lowEValue()
	) - make_zip_iterator(
			make_tuple(
					h_targetIDArray   .begin(),
					h_targetIndexArray.begin(),
					h_queryIDArray    .begin(),
					h_queryIndexArray .begin(),
					h_tHitLengthArray .begin(),
					h_qHitLengthArray .begin(),
					h_matchNumArray   .begin(),
					h_scoreArray      .begin(),
					h_evalueArray     .begin()
			)
	);
	targetIDArray    = h_targetIDArray;
	targetIndexArray = h_targetIndexArray;
	queryIDArray     = h_queryIDArray;
	queryIndexArray  = h_queryIndexArray;
	tHitLengthArray  = h_tHitLengthArray;
	qHitLengthArray  = h_qHitLengthArray;
	matchNumArray    = h_matchNumArray;
	scoreArray       = h_scoreArray;
	evalueArray      = h_evalueArray;
	/* resize */
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
		thrust::device_vector<double>& evalueArray) {
	time_attack::runLabeledWithSuffix(
			"  ...deleting hits which has low e-value", "...................finished.",
			[&] {
				removeLowEValueHits(
						cutOffEValue,
						targetIDArray,
						targetIndexArray,
						queryIDArray,
						queryIndexArray,
						tHitLengthArray,
						qHitLengthArray,
						matchNumArray,
						scoreArray,
						evalueArray);
			},
			[&](std::ostream& os, float /*ms*/) {
				os << targetIDArray.size() << " un-exact hits found." << std::endl;
			});
}

void removeTooShortHits(
		const CHostSetting& s,
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
	host_vector<int>    h_targetIDArray    = targetIDArray;
	host_vector<int>    h_targetIndexArray = targetIndexArray;
	host_vector<int>    h_queryIDArray     = queryIDArray;
	host_vector<int>    h_queryIndexArray  = queryIndexArray;
	host_vector<int>    h_tHitLengthArray  = tHitLengthArray;
	host_vector<int>    h_qHitLengthArray  = qHitLengthArray;
	host_vector<int>    h_matchNumArray    = matchNumArray;
	host_vector<int>    h_scoreArray       = scoreArray;
	host_vector<double> h_evalueArray      = evalueArray;
	/* remove */
	const int new_size = remove_if(
			make_zip_iterator(
					make_tuple(
							h_targetIDArray   .begin(),
							h_targetIndexArray.begin(),
							h_queryIDArray    .begin(),
							h_queryIndexArray .begin(),
							h_tHitLengthArray .begin(),
							h_qHitLengthArray .begin(),
							h_matchNumArray   .begin(),
							h_scoreArray      .begin(),
							h_evalueArray     .begin()
					)
			),
			make_zip_iterator(
					make_tuple(
							h_targetIDArray   .end(),
							h_targetIndexArray.end(),
							h_queryIDArray    .end(),
							h_queryIndexArray .end(),
							h_tHitLengthArray .end(),
							h_qHitLengthArray .end(),
							h_matchNumArray   .end(),
							h_scoreArray      .end(),
							h_evalueArray     .end()
					)
			),
			make_zip_iterator(
					make_tuple(
							make_constant_iterator(s.getLMerLength() + s.getStrideLength()),
							h_qHitLengthArray.begin()
					)
			),
			is_tooShort()
	) - make_zip_iterator(
			make_tuple(
					h_targetIDArray   .begin(),
					h_targetIndexArray.begin(),
					h_queryIDArray    .begin(),
					h_queryIndexArray .begin(),
					h_tHitLengthArray .begin(),
					h_qHitLengthArray .begin(),
					h_matchNumArray   .begin(),
					h_scoreArray      .begin(),
					h_evalueArray     .begin()
			)
	);
	targetIDArray    = h_targetIDArray;
	targetIndexArray = h_targetIndexArray;
	queryIDArray     = h_queryIDArray;
	queryIndexArray  = h_queryIndexArray;
	tHitLengthArray  = h_tHitLengthArray;
	qHitLengthArray  = h_qHitLengthArray;
	matchNumArray    = h_matchNumArray;
	scoreArray       = h_scoreArray;
	evalueArray      = h_evalueArray;
	/* resize */
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
		thrust::device_vector<double>& evalueArray) {
	time_attack::runLabeledWithSuffix(
			"  ...deleting too short hits", "...............................finished.",
			[&] {
				removeTooShortHits(
						s,
						targetIDArray,
						targetIndexArray,
						queryIDArray,
						queryIndexArray,
						tHitLengthArray,
						qHitLengthArray,
						matchNumArray,
						scoreArray,
						evalueArray);
			},
			[&](std::ostream& os, float /*ms*/) {
				os << targetIDArray.size() << " un-exact hits found." << std::endl;
			});
}
