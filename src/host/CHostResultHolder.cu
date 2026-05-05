#include "host/CHostResultHolder.cuh"

#include "util/utilResultSorting.cuh"
#include "util/time_attack.hpp"

#include <thrust/host_vector.h>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <thrust/sequence.h>

/*********************************** class function **********************************/

/* called at main() before CHostSchedular::search() was called */
CHostResultHolder::CHostResultHolder(const std::vector<CHostFASTA>& qFASTA) {
	for(std::vector<CHostFASTA>::const_iterator it = qFASTA.begin(); it != qFASTA.end(); ++it) {
		const std::string& seq = (*it).getLabel();
		queryLabelArray.push_back(seq);
		queryLabelArray.push_back(seq);
		queryStrandArray.push_back('+');
		queryStrandArray.push_back('-');
	}
}

/* called at CHostSchedular::search() */
void CHostResultHolder::addResult(
		const thrust::host_vector<int>& tIDArray,
		const thrust::host_vector<int>& tIndexArray,
		const thrust::host_vector<int>& qIDArray,
		const thrust::host_vector<int>& qIndexArray,
		const thrust::host_vector<int>& tLengthArray,
		const thrust::host_vector<int>& qLengthArray,
		const thrust::host_vector<int>& mNumArray,
		const thrust::host_vector<int>& sArray,
		const thrust::host_vector<double>& evalArray) {
	using namespace thrust;
	const int oldSize = targetIDArray.size();
	const int newSize = oldSize + tIDArray.size();

	targetIDArray   .resize(newSize);
	targetIndexArray.resize(newSize);
	queryIDArray    .resize(newSize);
	queryIndexArray .resize(newSize);
	tHitLengthArray .resize(newSize);
	qHitLengthArray .resize(newSize);
	matchNumArray   .resize(newSize);
	scoreArray      .resize(newSize);
	evalueArray     .resize(newSize);

	copy(tIDArray    .begin(), tIDArray    .end(), targetIDArray   .begin() + oldSize);
	copy(tIndexArray .begin(), tIndexArray .end(), targetIndexArray.begin() + oldSize);
	copy(qIDArray    .begin(), qIDArray    .end(), queryIDArray    .begin() + oldSize);
	copy(qIndexArray .begin(), qIndexArray .end(), queryIndexArray .begin() + oldSize);
	copy(tLengthArray.begin(), tLengthArray.end(), tHitLengthArray .begin() + oldSize);
	copy(qLengthArray.begin(), qLengthArray.end(), qHitLengthArray .begin() + oldSize);
	copy(mNumArray   .begin(), mNumArray   .end(), matchNumArray   .begin() + oldSize);
	copy(sArray      .begin(), sArray      .end(), scoreArray      .begin() + oldSize);
	copy(evalArray   .begin(), evalArray   .end(), evalueArray     .begin() + oldSize);
}

/* called at CHostSchedular::search() */
void CHostResultHolder::addLabel   (const CHostSeqList_target& targetList) {
	for(int i = targetLabelArray.size(); i < targetIDArray.size(); ++i) {
		targetLabelArray.push_back(targetList.getLabel(targetIDArray[i]));
	}
}

/* called at CHostSchedular::search() */
void CHostResultHolder::addStartIdx(const CHostSeqList_target& targetList) {
	for(int i = targetStartIdxArray.size(); i < targetIDArray.size(); ++i) {
		targetStartIdxArray.push_back(targetList.getStartIdx(targetIDArray[i]));
	}
}

/* called at main() after CHostSchedular::search() was called */
void CHostResultHolder::fixResult  (void) {
	using namespace thrust;

	device_vector<int> d_targetIDArray    = targetIDArray;
	device_vector<int> d_targetIndexArray = targetIndexArray;
	device_vector<int> d_queryIDArray     = queryIDArray;
	device_vector<int> d_queryIndexArray  = queryIndexArray;
	device_vector<int> d_tHitLengthArray  = tHitLengthArray;
	device_vector<int> d_qHitLengthArray  = qHitLengthArray;
	device_vector<int> d_matchNumArray    = matchNumArray;
	device_vector<int> d_scoreArray       = scoreArray;
	device_vector<double> d_evalueArray   = evalueArray;

	// CAUTION : change of the meaning of "targetIDArray".
	// See also: addLabel() and addStartIdx()
	sequence(d_targetIDArray.begin(), d_targetIDArray.end());

	time_attack::runLabeled(
			"\n  ...sorting results", ".......................................finished.",
			[&] {
				clast::sorting::resultSorting(
						d_targetIDArray,
						d_targetIndexArray,
						d_queryIDArray,
						d_queryIndexArray,
						d_tHitLengthArray,
						d_qHitLengthArray,
						d_matchNumArray,
						d_scoreArray,
						d_evalueArray);
			},
			time_attack::NewlineAfterCosts::No);

	targetIDArray    = d_targetIDArray;
	targetIndexArray = d_targetIndexArray;
	queryIDArray     = d_queryIDArray;
	queryIndexArray  = d_queryIndexArray;
	tHitLengthArray  = d_tHitLengthArray;
	qHitLengthArray  = d_qHitLengthArray;
	matchNumArray    = d_matchNumArray;
	scoreArray       = d_scoreArray;
	evalueArray      = d_evalueArray;
}

/* called at main() after CHostSchedular::search() was called */
void CHostResultHolder::printResultToFile(
		int numberOfOutput,
		const std::string& outputFile) const {
	clast::result::printResultToFile(
			numberOfOutput, outputFile,
			queryLabelArray,    queryStrandArray,
			targetLabelArray,   targetStartIdxArray,
			targetIDArray,      targetIndexArray,
			queryIDArray,       queryIndexArray,
			tHitLengthArray,    qHitLengthArray,
			matchNumArray,      scoreArray,
			evalueArray);
}

void CHostResultHolder::printResult(
		const int numberOfOutput,
		const std::string& outputFile) const {
	time_attack::runLabeled(
			"  ...print result to disc", "..................................finished.",
			[&] { printResultToFile(numberOfOutput, outputFile); });
}
