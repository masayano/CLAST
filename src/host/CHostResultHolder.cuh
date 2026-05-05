#ifndef C_HOST_RESULT_HOLDER_CUH_
#define C_HOST_RESULT_HOLDER_CUH_

#include "host/seq/query.cuh"
#include "host/seq/target.cuh"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <thrust/host_vector.h>

namespace clast::result {

struct PrevRow {
	std::string queryLabel;
	int         queryIndex  = {};
	char        queryStrand = {};
	std::string targetLabel;
	int         targetIndex = {};
	int         tHitLength  = {};
	int         qHitLength  = {};
	int         matchNum    = {};
	int         score       = {};
	double      eValue      = {};

	bool matches(
			const std::string& qLabel, int qIdx, char qStrand,
			const std::string& tLabel, int tIdx,
			int tLen, int qLen, int mNum, int sc, double ev) const {
		return queryLabel  == qLabel  &&
		       queryIndex  == qIdx   &&
		       queryStrand == qStrand &&
		       targetLabel == tLabel  &&
		       targetIndex == tIdx   &&
		       tHitLength  == tLen   &&
		       qHitLength  == qLen   &&
		       matchNum    == mNum   &&
		       score       == sc     &&
		       eValue      == ev;
	}

	void set(
			const std::string& qLabel, int qIdx, char qStrand,
			const std::string& tLabel, int tIdx,
			int tLen, int qLen, int mNum, int sc, double ev) {
		queryLabel  = qLabel;
		queryIndex  = qIdx;
		queryStrand = qStrand;
		targetLabel = tLabel;
		targetIndex = tIdx;
		tHitLength  = tLen;
		qHitLength  = qLen;
		matchNum    = mNum;
		score       = sc;
		eValue      = ev;
	}
};

inline void printResultToFile(
		int numberOfOutput,
		const std::string& outputFile,
		const std::vector<std::string>& queryLabelArray,
		const std::vector<char>&        queryStrandArray,
		const std::vector<std::string>& targetLabelArray,
		const std::vector<int>&         targetStartIdxArray,
		const thrust::host_vector<int>& targetIDArray,
		const thrust::host_vector<int>& targetIndexArray,
		const thrust::host_vector<int>& queryIDArray,
		const thrust::host_vector<int>& queryIndexArray,
		const thrust::host_vector<int>& tHitLengthArray,
		const thrust::host_vector<int>& qHitLengthArray,
		const thrust::host_vector<int>& matchNumArray,
		const thrust::host_vector<int>& scoreArray,
		const thrust::host_vector<double>& evalueArray) {
	std::stringstream filename;
	filename << outputFile;

	std::ofstream ofs;
	ofs.open(filename.str().c_str(), std::ios::app);

	PrevRow prev;
	int count = 0;
	int printedHitsCounter = 0;
	for(int i = 0; i < queryIDArray.size(); ++i) {
		const std::string& queryLabel = queryLabelArray[queryIDArray[i]];
		if(prev.queryLabel != queryLabel) { count = 0; }
		if(
			(numberOfOutput == -1) ||          // unlimited.
			(prev.queryLabel != queryLabel) || // top hit.
			(count < numberOfOutput)           // other hit.
		) {
			const std::string& targetLabel = targetLabelArray[targetIDArray[i]];
			const int    queryIndex  = queryIndexArray[i];
			const char   queryStrand = queryStrandArray[queryIDArray[i]];
			const int    targetIndex = targetIndexArray[i] + targetStartIdxArray[targetIDArray[i]];
			const int    tHitLength  = tHitLengthArray[i];
			const int    qHitLength  = qHitLengthArray[i];
			const int    matchNum    = matchNumArray[i];
			const int    score       = scoreArray[i];
			const double eValue      = evalueArray[i];
			if(!prev.matches(queryLabel, queryIndex, queryStrand,
			                 targetLabel, targetIndex,
			                 tHitLength, qHitLength, matchNum, score, eValue)) {
				ofs	<< queryLabel
					<< "\t"
					<< queryIndex
					<< "\t"
					<< qHitLength
					<< "\t"
					<< queryStrand
					<< "\t"
					<< targetLabel
					<< "\t"
					<< targetIndex
					<< "\t"
					<< tHitLength
					<< "\t"
					<< matchNum << "(" << static_cast<double>(matchNum*100)/qHitLength << "%)"
					<< "\t"
					<< score
					<< "\t"
					<< eValue
					<< std::endl;
				++count;
				++printedHitsCounter;
				prev.set(queryLabel, queryIndex, queryStrand,
				         targetLabel, targetIndex,
				         tHitLength, qHitLength, matchNum, score, eValue);
			}
		}
	}
	std::cout << " " << printedHitsCounter << " hits has printed." << std::endl;
}

} // namespace clast::result

class CHostResultHolder {
	std::vector<std::string> queryLabelArray;
	std::vector<char> queryStrandArray;
	std::vector<std::string> targetLabelArray;
	std::vector<int> targetStartIdxArray;
	thrust::host_vector<int> targetIDArray;
	thrust::host_vector<int> targetIndexArray;
	thrust::host_vector<int> queryIDArray;
	thrust::host_vector<int> queryIndexArray;
	thrust::host_vector<int> tHitLengthArray;
	thrust::host_vector<int> qHitLengthArray;
	thrust::host_vector<int> matchNumArray;
	thrust::host_vector<int> scoreArray;
	thrust::host_vector<double> evalueArray;
public:
	CHostResultHolder(const std::vector<CHostFASTA>& qFASTA);
	void addResult(
			const thrust::host_vector<int>& tIDArray,
			const thrust::host_vector<int>& tIndexArray,
			const thrust::host_vector<int>& qIDArray,
			const thrust::host_vector<int>& qIndexArray,
			const thrust::host_vector<int>& tLengthArray,
			const thrust::host_vector<int>& qLengthArray,
			const thrust::host_vector<int>& mNumArray,
			const thrust::host_vector<int>& sArray,
			const thrust::host_vector<double>& evalArray);
	void addLabel   (const CHostSeqList_target& targetList);
	void addStartIdx(const CHostSeqList_target& targetList);
	void fixResult  (void);
	void printResult(
			const int numberOfOutput,
			const std::string& outputFile) const;
private:
	void printResultToFile(
			int numberOfOutput,
			const std::string& outputFile) const;
};

#endif
