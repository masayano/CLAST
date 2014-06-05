#include "CHostResultHolder.cuh"

#include "utilResultSorting.cuh"

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

	#ifdef TIME_ATTACK
		float elapsed_time_ms=0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate( &start );
		cudaEventCreate( &stop  );
		cudaEventRecord( start, 0 );
		std::cout << std::endl << "  ...sorting results";
	#endif /* TIME_ATTACK */
	resultSorting(
			d_targetIDArray,
			d_targetIndexArray,
			d_queryIDArray,
			d_queryIndexArray,
			d_tHitLengthArray,
			d_qHitLengthArray,
			d_matchNumArray,
			d_scoreArray,
			d_evalueArray);
	#ifdef TIME_ATTACK
		std::cout << ".......................................finished.";
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		std::cout
				<< " (costs " << elapsed_time_ms << "ms)";
	#endif /* TIME_ATTACK */

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
void CHostResultHolder::printResult(
		const int numberOfOutput,
		const std::string& outputFile) const {
	#ifdef TIME_ATTACK
		float elapsed_time_ms=0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate( &start );
		cudaEventCreate( &stop  );
		cudaEventRecord( start, 0 );
		std::cout << "  ...print result to disc";
	#endif /* TIME_ATTACK */
	std::stringstream filename;
	filename << outputFile;

	std::ofstream ofs;
	ofs.open(filename.str().c_str(), std::ios::app);

	std::string preQueryLabel;
	int         preQueryIndex;
	char        preQueryStrand;
	std::string preTargetLabel;
	int         preTargetIndex;
	int         preTHitLength;
	int         preQHitLength;
	int         preMatchNum;
	int         preScore;
	double      preEValue;

	int          queryIndex;
	char         queryStrand;
	int          targetIndex;
	int          tHitLength;
	int          qHitLength;
	int          matchNum;
	int          score;
	double       eValue;

	int count = 0;
	int printedHitsCounter = 0;
	for(int i = 0; i < queryIDArray.size(); ++i) {
		const std::string& queryLabel = queryLabelArray[queryIDArray[i]];
		if(preQueryLabel != queryLabel) { count = 0; }
		if(
			(numberOfOutput == -1) ||        // unlimited.
			(preQueryLabel != queryLabel) || // top hit.
			(count < numberOfOutput)         // other hit.
		) {
			const std::string& targetLabel = targetLabelArray[targetIDArray[i]];
			queryIndex  = queryIndexArray[i];
			queryStrand = queryStrandArray[queryIDArray[i]];
			targetIndex = targetIndexArray[i] + targetStartIdxArray[targetIDArray[i]];
			tHitLength  = tHitLengthArray[i];
			qHitLength  = qHitLengthArray[i];
			matchNum    = matchNumArray[i];
			score       = scoreArray[i];
			eValue      = evalueArray[i];
			if(
				(preQueryLabel  != queryLabel ) ||
				(preQueryIndex  != queryIndex ) ||
				(preQueryStrand != queryStrand) ||
				(preTargetLabel != targetLabel) ||
				(preTargetIndex != targetIndex) ||
				(preTHitLength  != tHitLength ) ||
				(preQHitLength  != qHitLength ) ||
				(preMatchNum    != matchNum   ) ||
				(preScore       != score      ) ||
				(preEValue      != eValue     )
			) {
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
				preQueryLabel  = queryLabel;
				preQueryIndex  = queryIndex;
				preQueryStrand = queryStrand;
				preTargetLabel = targetLabel;
				preTargetIndex = targetIndex;
				preTHitLength  = tHitLength;
				preQHitLength  = qHitLength;
				preMatchNum    = matchNum;
				preScore       = score;
				preEValue      = eValue;
			}
		}
	}
	std::cout << " " << printedHitsCounter << " hits has printed." << std::endl;

	#ifdef TIME_ATTACK
		std::cout << "..................................finished.";
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		std::cout << " (costs " << elapsed_time_ms << "ms)" << std::endl;
	#endif /* TIME_ATTACK */
}
