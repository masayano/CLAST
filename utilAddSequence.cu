#include "utilAddSequence.cuh"

#include "utilReverseSeq.hpp"
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

	host_vector<int> newIdxArray(seqLength + jointLength);
	sequence(newIdxArray.begin(), newIdxArray.end());

	if(indexArray.empty()){
		indexArray.assign(newIdxArray.begin(), newIdxArray.end());
		IDArray   .assign(seqLength + jointLength, 0);
		baseArray .assign(FASTAseq.begin(), FASTAseq.end());
	} else {
		indexArray.insert(
				indexArray.end(),
				newIdxArray.begin(),
				newIdxArray.end()
		);
		IDArray.insert(
				IDArray.end(),
				seqLength + jointLength,
				IDArray.back() + 1
		);
		baseArray.insert(
				baseArray.end(),
				FASTAseq .begin(),
				FASTAseq .end()
		);
	}
	/* overlap (length : jointLength) */
	baseArray.insert(
			baseArray.end(),
			FASTAseq .begin(),
			FASTAseq .begin() + jointLength
	);
}
