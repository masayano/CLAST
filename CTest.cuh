#ifndef C_TEST_CUH_
#define C_TEST_CUH_

#include "CDeviceSeqList.cuh"
#include "CDeviceHitList.cuh"

#include <fstream>
#include <string>

#include <thrust/host_vector.h>

class CTest {
	static int seed;
	static std::ofstream logger;
	CTest(void);
public:
	static void openLogger(void);

	static std::string generateSequence(const int size);

	static std::string randomQuote(
			const std::vector<std::string>& str,
			const int length);

	static std::string insertError(
			const int mismatchNum,
			std::string& str);

	static std::string insertion(std::string& str);

	static std::string deletion(std::string& str);

	static void printAddedSeq(const CDeviceSeqList* s);

	static void printNonoverlappedHashIndex(
			const int strideLength,
			const thrust::host_vector<char> base,
			const thrust::host_vector<long> hashIndex);

	static void printOverlappedHashIndex(
			const thrust::host_vector<char> base,
			const thrust::host_vector<long> hashIndex);

	static void printHashTable(
			const thrust::host_vector<int> gateway,
			const thrust::host_vector<int> indexArray);

	static void printQueryToTarget(
			const thrust::host_vector<int> target_IDArray,
			const thrust::host_vector<int> target_indexArray,
			const thrust::host_vector<int> query_IDArray,
			const thrust::host_vector<int> query_indexArray);

	static void printIsolatedHit(
			const thrust::host_vector<int> target_IDArray,
			const thrust::host_vector<int> target_indexArray,
			const thrust::host_vector<int> query_IDArray,
			const thrust::host_vector<int> query_indexArray,
			const thrust::host_vector<int> hitLengthArray);

	static void printIsolatedHit(
			const thrust::host_vector<int> target_IDArray,
			const thrust::host_vector<int> target_indexArray,
			const thrust::host_vector<int> query_IDArray,
			const thrust::host_vector<int> query_indexArray,
			const thrust::host_vector<int> hitLengthArray,
			const thrust::host_vector<int> gapNumArray,
			const thrust::host_vector<int> mismatchNumArray);

	static void printResult(
			const thrust::host_vector<int> targetIDArray,
			const thrust::host_vector<int> targetIndexArray,
			const thrust::host_vector<int> queryIDArray,
			const thrust::host_vector<int> gapNumArray,
			const thrust::host_vector<int> mismatchNumArray);

	static void printPairedHits(
			const thrust::host_vector<int> paired_0_targetIDArray,
			const thrust::host_vector<int> paired_0_targetIndexArray,
			const thrust::host_vector<int> paired_0_queryIDArray,
			const thrust::host_vector<int> paired_0_queryIndexArray,
			const thrust::host_vector<int> paired_0_hitLengthArray,
			const thrust::host_vector<int> paired_1_targetIDArray,
			const thrust::host_vector<int> paired_1_targetIndexArray,
			const thrust::host_vector<int> paired_1_queryIDArray,
			const thrust::host_vector<int> paired_1_queryIndexArray,
			const thrust::host_vector<int> paired_1_hitLengthArray);
};

#endif
