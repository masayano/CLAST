#include "device/hit/deleteBadHits.cuh"

#include "util/common.hpp"
#include "util/time_attack.hpp"

#include <iostream>

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
	clast::hit::removeLowEValue(
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
	clast::hit::removeTooShort(
			s.getLMerLength() + s.getStrideLength(),
			targetIDArray,
			targetIndexArray,
			queryIDArray,
			queryIndexArray,
			tHitLengthArray,
			qHitLengthArray,
			matchNumArray,
			scoreArray,
			evalueArray);
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
