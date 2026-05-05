#include "device/hit/CDeviceHitList.cuh"

#include "device/hit/alignmentHits.cuh"
#include "device/hit/createRawSeedList.cuh"
#include "device/hit/deleteBadHits.cuh"
#include "device/hit/deleteDuplicateSeeds.cuh"
#include "device/hit/deleteIsolateSeeds.cuh"
#include "device/hit/deleteSeedsOnSequenceBoundary.cuh"
#include "device/hit/sortSeeds.cuh"
#include "util/utilResultSorting.cuh"

#include "kernel/krnlCalculateEvalue.cuh"
#include "kernel/krnlAlignment.cuh"

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include "util/common.hpp"
#include "util/time_attack.hpp"
#include <iostream>

#ifdef MODE_TEST
#include "test_support/CTest.cuh"
#endif /* MODE_TEST */

/*************************************** private **************************************/

void deleteHits_duplicateResult(
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
			"  ...deleting duplicate hits", "...............................finished.",
			[&] {
				clast::hit::removeDuplicateResultHits(
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
				os << targetIDArray.size() << " hits found." << std::endl;
			});
}

/********************************* non class functions ***********************************/

void appendUnExactMatchResults(
		const thrust::device_vector<int>& add_targetIDArray,
		const thrust::device_vector<int>& add_targetIndexArray,
		const thrust::device_vector<int>& add_queryIDArray,
		const thrust::device_vector<int>& add_queryIndexArray,
		const thrust::device_vector<int>& add_tHitLengthArray,
		const thrust::device_vector<int>& add_qHitLengthArray,
		const thrust::device_vector<int>& add_matchNumArray,
		const thrust::device_vector<int>& add_scoreArray,
		const thrust::device_vector<double>& add_evalueArray,
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
	const int oldResultSize = targetIDArray.size();
	const int newResultSize = oldResultSize + add_targetIDArray.size();
	targetIDArray.resize(newResultSize);
	targetIndexArray.resize(newResultSize);
	queryIDArray.resize(newResultSize);
	queryIndexArray.resize(newResultSize);
	tHitLengthArray.resize(newResultSize);
	qHitLengthArray.resize(newResultSize);
	matchNumArray.resize(newResultSize);
	scoreArray.resize(newResultSize);
	evalueArray.resize(newResultSize);
	copy(
			add_targetIDArray.begin(),
			add_targetIDArray.end(),
			targetIDArray.begin() + oldResultSize);
	copy(
			add_targetIndexArray.begin(),
			add_targetIndexArray.end(),
			targetIndexArray.begin() + oldResultSize);
	copy(
			add_queryIDArray.begin(),
			add_queryIDArray.end(),
			queryIDArray.begin() + oldResultSize);
	copy(
			add_queryIndexArray.begin(),
			add_queryIndexArray.end(),
			queryIndexArray.begin() + oldResultSize);
	copy(
			add_tHitLengthArray.begin(),
			add_tHitLengthArray.end(),
			tHitLengthArray.begin() + oldResultSize);
	copy(
			add_qHitLengthArray.begin(),
			add_qHitLengthArray.end(),
			qHitLengthArray.begin() + oldResultSize);
	copy(
			add_matchNumArray.begin(),
			add_matchNumArray.end(),
			matchNumArray.begin() + oldResultSize);
	copy(
			add_scoreArray.begin(),
			add_scoreArray.end(),
			scoreArray.begin() + oldResultSize);
	copy(
			add_evalueArray.begin(),
			add_evalueArray.end(),
			evalueArray.begin() + oldResultSize);
}

void addResult(
		const thrust::device_vector<int>& add_targetIDArray,
		const thrust::device_vector<int>& add_targetIndexArray,
		const thrust::device_vector<int>& add_queryIDArray,
		const thrust::device_vector<int>& add_queryIndexArray,
		const thrust::device_vector<int>& add_tHitLengthArray,
		const thrust::device_vector<int>& add_qHitLengthArray,
		const thrust::device_vector<int>& add_matchNumArray,
		const thrust::device_vector<int>& add_scoreArray,
		const thrust::device_vector<double>& add_evalueArray,
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
			"  ...writing un-exact matches", "..............................finished.",
			[&] {
				appendUnExactMatchResults(
						add_targetIDArray,
						add_targetIndexArray,
						add_queryIDArray,
						add_queryIndexArray,
						add_tHitLengthArray,
						add_qHitLengthArray,
						add_matchNumArray,
						add_scoreArray,
						add_evalueArray,
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
				os << add_targetIDArray.size() << " un-exact hits found." << std::endl;
			});
}

/************************************* class functions **************************************/
CDeviceHitList::CDeviceHitList(
		const CHostSetting& s,
		const CDeviceHashTable& h,
		const CDeviceSeqList_query& q,
		const int t_begin,
		const int q_begin) {
	using namespace thrust;

	/* prepare seed hit list */
	device_vector<int> seed_targetIDArray;
	device_vector<int> seed_targetIndexArray;
	device_vector<int> seed_queryIDArray;
	device_vector<int> seed_queryIndexArray;

	createRawSeedList(
			s,
			h,
			q,
			t_begin,
			q_begin,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
	deleteSeedsOnSequenceBoundary(
			s,
			h,
			q,
			t_begin,
			q_begin,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);

	sortSeeds(
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
	deletingIsolateSeeds(
			s.getAllowableWidth(),
			s.getAllowableGap(),
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);
	deleteDuplicateSeeds(
			s,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray);

	/* prepare hit list array */
	device_vector<int> seed_tLengthArray (seed_targetIDArray.size(), s.getLMerLength());
	device_vector<int> seed_qLengthArray (seed_targetIDArray.size(), s.getLMerLength());
	device_vector<int> seed_matchNumArray(seed_targetIDArray.size(), s.getLMerLength());
	device_vector<int> seed_scoreArray   (seed_targetIDArray.size(), s.getLMerLength() * MATCH_POINT);

	alignmentHits(
			s,
			h,
			q,
			t_begin,
			q_begin,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray,
			seed_tLengthArray,
			seed_qLengthArray,
			seed_matchNumArray,
			seed_scoreArray);

	/* prepare calculation of E-value */
	thrust::device_vector<double> seed_evalueArray(seed_targetIDArray.size());

	time_attack::runLabeledWithSuffix(
			"  ...calculating E-value", "...................................finished.",
			[&] {
				const int blockDim_x = 256;
				const int seedAlignmentNum = seed_targetIDArray.size();
				calculateEvalue<<<(seedAlignmentNum/blockDim_x)+1, blockDim_x>>>(
						q_begin,
						seedAlignmentNum,
						s.getTotalDatabaseSize(),
						s.getK(),
						s.getLambda(),
						raw_pointer_cast( &*q.getLengthArray().begin() ),
						raw_pointer_cast( &*seed_queryIDArray .begin() ),
						raw_pointer_cast( &*seed_scoreArray   .begin() ),
						raw_pointer_cast( &*seed_evalueArray  .begin() )
				);
			},
			[&](std::ostream& os, float /*ms*/) {
				os << seed_targetIDArray.size() << " hits found." << std::endl;
			});

	if(s.getCutOff() != -1) {
		deleteHits_lowEValue(
				s.getCutOff(),
				seed_targetIDArray,
				seed_targetIndexArray,
				seed_queryIDArray,
				seed_queryIndexArray,
				seed_tLengthArray,
				seed_qLengthArray,
				seed_matchNumArray,
				seed_scoreArray,
				seed_evalueArray);
	}

	deleteHits_tooShort(
			s,
			seed_targetIDArray,
			seed_targetIndexArray,
			seed_queryIDArray,
			seed_queryIndexArray,
			seed_tLengthArray,
			seed_qLengthArray,
			seed_matchNumArray,
			seed_scoreArray,
			seed_evalueArray);

	if(seed_targetIDArray.size() != 0) {
		addResult(
				seed_targetIDArray,
				seed_targetIndexArray,
				seed_queryIDArray,
				seed_queryIndexArray,
				seed_tLengthArray,
				seed_qLengthArray,
				seed_matchNumArray,
				seed_scoreArray,
				seed_evalueArray,
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

	time_attack::runLabeledWithSuffix(
			"  ...sorting results", ".......................................finished.",
			[&] {
				clast::sorting::resultSorting(
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
				os << targetIDArray.size() << " total hits found." << std::endl;
			});

	deleteHits_duplicateResult(
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

void CDeviceHitList::getResult(CHostResultHolder& holder) {
	time_attack::runLabeled(
			"  ...Record result to host", ".................................finished.",
			[&] {
				holder.addResult(
						targetIDArray,
						targetIndexArray,
						queryIDArray,
						queryIndexArray,
						tHitLengthArray,
						qHitLengthArray,
						matchNumArray,
						scoreArray,
						evalueArray);
			});
}
