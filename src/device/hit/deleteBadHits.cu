#include "device/hit/deleteBadHits.cuh"

#include "util/common.hpp"
#include "util/time_attack.hpp"

#include <iostream>
#include <thrust/host_vector.h>

void removeLowEValueHitsImpl(
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
	clast::hit::removeLowEValue(
			cutOffEValue,
			h_targetIDArray,
			h_targetIndexArray,
			h_queryIDArray,
			h_queryIndexArray,
			h_tHitLengthArray,
			h_qHitLengthArray,
			h_matchNumArray,
			h_scoreArray,
			h_evalueArray);
	targetIDArray    = h_targetIDArray;
	targetIndexArray = h_targetIndexArray;
	queryIDArray     = h_queryIDArray;
	queryIndexArray  = h_queryIndexArray;
	tHitLengthArray  = h_tHitLengthArray;
	qHitLengthArray  = h_qHitLengthArray;
	matchNumArray    = h_matchNumArray;
	scoreArray       = h_scoreArray;
	evalueArray      = h_evalueArray;
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
				removeLowEValueHitsImpl(
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

void removeTooShortHitsImpl(
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
	clast::hit::removeTooShort(
			s.getLMerLength() + s.getStrideLength(),
			h_targetIDArray,
			h_targetIndexArray,
			h_queryIDArray,
			h_queryIndexArray,
			h_tHitLengthArray,
			h_qHitLengthArray,
			h_matchNumArray,
			h_scoreArray,
			h_evalueArray);
	targetIDArray    = h_targetIDArray;
	targetIndexArray = h_targetIndexArray;
	queryIDArray     = h_queryIDArray;
	queryIndexArray  = h_queryIndexArray;
	tHitLengthArray  = h_tHitLengthArray;
	qHitLengthArray  = h_qHitLengthArray;
	matchNumArray    = h_matchNumArray;
	scoreArray       = h_scoreArray;
	evalueArray      = h_evalueArray;
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
				removeTooShortHitsImpl(
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
