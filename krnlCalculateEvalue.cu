#include "krnlCalculateEvalue.cuh"

#include "common.hpp"

__global__ void calculateEvalue(
		const int q_begin,
		const int matchSize,
		const double totalDatabaseSize,
		const double K,
		const double lambda,
		const int* queryLengthArray,
		const int* queryIDArray,
		const int* scoreArray,
		double* evalueArray) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < matchSize) {
		const int queryLength = queryLengthArray[queryIDArray[idx] - q_begin];
		const int score       = scoreArray      [idx];

		evalueArray[idx] = K * totalDatabaseSize * queryLength * exp(-lambda * score);
	}
}
