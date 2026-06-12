#include "util/utilAddSequence.cuh"

#include "util/utilReverseSeq.hpp"
#include <thrust/sequence.h>

void addSequence(
		const int seqLength,
		const int lMerLength,
		const std::string& FASTAseq,
		thrust::host_vector<int>&  indexArray,
		thrust::host_vector<int>&  IDArray,
		thrust::host_vector<char>& baseArray) {
	using namespace thrust;

	const int jointLength = lMerLength - 1;
	const int addLength = seqLength + jointLength;
	const int newID = indexArray.empty() ? 0 : IDArray.back() + 1;
	const size_t oldSize = indexArray.size();

	indexArray.resize(oldSize + addLength);
	sequence(indexArray.begin() + oldSize, indexArray.end());

	IDArray.resize(oldSize + addLength, newID);

	baseArray.insert(
			baseArray.end(),
			FASTAseq .begin(),
			FASTAseq .end()
	);
	/* overlap (length : jointLength) */
	baseArray.insert(
			baseArray.end(),
			FASTAseq .begin(),
			FASTAseq .begin() + jointLength
	);
}
