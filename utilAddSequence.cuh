#ifndef UTIL_ADD_SEQUENCE_HPP_
#define UTIL_ADD_SEQUENCE_HPP

#include <string>
#include <thrust/host_vector.h>

void addSequence(
		const int seqLength,
		const int lMerLength,
		const std::string& FASTAseq,
		thrust::host_vector<int>&  indexArray,
		thrust::host_vector<int>&  IDArray,
		thrust::host_vector<char>& baseArray);

#endif /* UTIL_ADD_SEQUENCE_HPP */
