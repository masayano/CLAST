#include "CTest.cuh"
#include "common.hpp"

#include <cstdlib>
#include <ctime>
#include <sstream>

int           CTest::seed = 0;
std::ofstream CTest::logger;

void CTest::openLogger(void) {
	std::stringstream filename;
	filename << time(0) << ".log";
	logger.open(filename.str().c_str());
}

void CTest::printAddedSeq(const CDeviceSeqList* s) {
	using namespace thrust;

	const host_vector<int>&  indexArray = s->getIndexArray();
	const host_vector<int>&  IDArray	   = s->getIDArray();
	const host_vector<char>& baseArray  = s->getBaseArray();

	for(int i = 0, j = 0; i < IDArray.size(); ++i) {
		if(indexArray[i] == 0) {
			logger << std::endl << "--- " << j++ << " ----" << std::endl;
		}
		logger << baseArray[i] << "(" << IDArray[i] << "/" << indexArray[i] << ") ";
	}
	logger << std::endl;
}

void CTest::printNonoverlappedHashIndex(
		const int strideLength,
		const thrust::host_vector<char> base,
		const thrust::host_vector<long> hashIndex) {
	logger << std::endl << "---- index,base,2bitbase,hash-index ----" << std::endl;
	thrust::host_vector<long>::const_iterator iterHashIndex = hashIndex.begin();
	for(int i = 0; i < base.size(); ++i) {
		char bitseq;
		switch(base[i]) {
			case 'A': bitseq = 0; break;
			case 'C': bitseq = 1; break;
			case 'G': bitseq = 2; break;
			case 'T': bitseq = 3; break;
		}

		if((i % strideLength == 0) && (iterHashIndex != hashIndex.end())) {
			logger	<< i
					<< " "
					<< base[i]
					<< " -> "
					<< (std::size_t)bitseq
					<< " ("
					<< *iterHashIndex++
					<< ")"
					<< std::endl;
		} else {
			logger	<< i
					<< " "
					<< base[i]
					<< " -> "
					<< (std::size_t)bitseq
					<< " ( - )"	
					<< std::endl;
		}
	}
}

void CTest::printOverlappedHashIndex(
		const thrust::host_vector<char> base,
		const thrust::host_vector<long> hashIndex) {
	logger << std::endl << "---- index,base,2bitbase,hash-index ----" << std::endl;
	thrust::host_vector<long>::const_iterator iterHashIndex = hashIndex.begin();
	for(int i = 0; i < base.size(); ++i) {
		char bitseq;
		switch(base[i]) {
			case 'A': bitseq = 0; break;
			case 'C': bitseq = 1; break;
			case 'G': bitseq = 2; break;
			case 'T': bitseq = 3; break;
		}

		if(iterHashIndex != hashIndex.end()) {
			logger	<< i
					<< " "
					<< base[i]
					<< " -> "
					<< (std::size_t)bitseq
					<< " ("
					<< *iterHashIndex++
					<< ")"
					<< std::endl;
		} else {
			logger	<< i
					<< " "
					<< base[i]
					<< " -> "
					<< (std::size_t)bitseq
					<< " ( - )"	
					<< std::endl; 
		}
	}
}

void CTest::printHashTable(
		const thrust::host_vector<int> gateway,
		const thrust::host_vector<int> indexArray) {
	logger << std::endl << "---- hash table ----" << std::endl;
	for(int i = 0, j=0; i < indexArray.size(); ++i) {
		while(i == gateway[j]) {
			logger << std::endl << " *** cell index : " << j++ << " ***" << std::endl;
		}
		logger << indexArray[i] << " ";
	}
	logger << std::endl;
}

void CTest::printQueryToTarget(
		const thrust::host_vector<int> target_IDArray,
		const thrust::host_vector<int> target_indexArray,
		const thrust::host_vector<int> query_IDArray,
		const thrust::host_vector<int> query_indexArray) {
	logger << " ---- (queryID/queryIndex)->(targetID/targetIndex) qIdx-tIdx ---- " << std::endl;
	for(int i = 0; i < query_IDArray.size(); ++i) {
		logger	<< " ("
				<< query_IDArray[i]
				<< "/" 
				<< query_indexArray[i]
				<< ")->("
				<< target_IDArray[i]
				<< "/"
				<< target_indexArray[i]
				<< ") "
				<< (query_indexArray[i] - target_indexArray[i])
				<< std::endl;
	}
}

void CTest::printIsolatedHit(
		const thrust::host_vector<int> target_IDArray,
		const thrust::host_vector<int> target_indexArray,
		const thrust::host_vector<int> query_IDArray,
		const thrust::host_vector<int> query_indexArray,
		const thrust::host_vector<int> hitLengthArray) {
	logger << " ---- (queryID/queryIndex)->(targetID/targetIndex) qIdx-tIdx(hit length) ---- " << std::endl;
	for(int i = 0; i < query_IDArray.size(); ++i) {
		logger	<< " ("
				<< query_IDArray[i]
				<< "/" 
				<< query_indexArray[i]
				<< ")->("
				<< target_IDArray[i]
				<< "/"
				<< target_indexArray[i]
				<< ") "
				<< (query_indexArray[i] - target_indexArray[i])
				<< " ("
				<< hitLengthArray[i]
				<< ")"
				<< std::endl;
	}
}

void CTest::printIsolatedHit(
		const thrust::host_vector<int> target_IDArray,
		const thrust::host_vector<int> target_indexArray,
		const thrust::host_vector<int> query_IDArray,
		const thrust::host_vector<int> query_indexArray,
		const thrust::host_vector<int> hitLengthArray,
		const thrust::host_vector<int> gapNumArray,
		const thrust::host_vector<int> mismatchNumArray) {
	logger << " ---- (queryID/queryIndex)->(targetID/targetIndex) qIdx-tIdx(hit length) gap/mismatch ---- " << std::endl;
	for(int i = 0; i < query_IDArray.size(); ++i) {
		logger	<< " ("
				<< query_IDArray[i]
				<< "/" 
				<< query_indexArray[i]
				<< ")->("
				<< target_IDArray[i]
				<< "/"
				<< target_indexArray[i]
				<< ") "
				<< (query_indexArray[i] - target_indexArray[i])
				<< " ("
				<< hitLengthArray[i]
				<< ") "
				<< gapNumArray[i]
				<< "/"
				<< mismatchNumArray[i]
				<< std::endl;
	}
}

void CTest::printResult(
			const thrust::host_vector<int> targetIDArray,
			const thrust::host_vector<int> targetIndexArray,
			const thrust::host_vector<int> queryIDArray,
			const thrust::host_vector<int> gapNumArray,
			const thrust::host_vector<int> mismatchNumArray) {
	logger << " ---- queryID->(targetID/targetIndex) mismatchNum & gapNum ---- " << std::endl;
	for(int i = 0; i < queryIDArray.size(); ++i) {
		logger	<< queryIDArray[i]
				<< "->("
				<< targetIDArray[i]
				<< "/"
				<< targetIndexArray[i]
				<< ") "
				<< mismatchNumArray[i]
				<< " & "
				<< gapNumArray[i]
				<< std::endl;
	}
}

void CTest::printPairedHits(
		const thrust::host_vector<int> paired_0_targetIDArray,
		const thrust::host_vector<int> paired_0_targetIndexArray,
		const thrust::host_vector<int> paired_0_queryIDArray,
		const thrust::host_vector<int> paired_0_queryIndexArray,
		const thrust::host_vector<int> paired_0_hitLengthArray,
		const thrust::host_vector<int> paired_1_targetIDArray,
		const thrust::host_vector<int> paired_1_targetIndexArray,
		const thrust::host_vector<int> paired_1_queryIDArray,
		const thrust::host_vector<int> paired_1_queryIndexArray,
		const thrust::host_vector<int> paired_1_hitLengthArray) {
	logger << " ---- (tID0/tIdx0) & (qID0/qIdx0) length0 + (tID1/tIdx1) & (qID1/qIdx1) length1 ---- " << std::endl;
	for(int i = 0; i < paired_0_targetIDArray.size(); ++i) {
		logger	<< " ("
				<< paired_0_targetIDArray[i]
				<< "/"
				<< paired_0_targetIndexArray[i]
				<< ") & ("
				<< paired_0_queryIDArray[i]
				<< "/"
				<< paired_0_queryIndexArray[i]
				<< ") "
				<< paired_0_hitLengthArray[i]
				<< " + ("
				<< paired_1_targetIDArray[i]
				<< "/"
				<< paired_1_targetIndexArray[i]
				<< ") & ("
				<< paired_1_queryIDArray[i]
				<< "/"
				<< paired_1_queryIndexArray[i]
				<< ") "
				<< paired_1_hitLengthArray[i]
				<< std::endl;
	}
}
