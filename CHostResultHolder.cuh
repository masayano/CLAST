#ifndef C_HOST_RESULT_HOLDER_CUH_
#define C_HOST_RESULT_HOLDER_CUH_

#include "CHostSeqList_query.cuh"
#include "CHostSeqList_target.cuh"

#include <string>
#include <vector>

#include <thrust/host_vector.h>

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
};

#endif
