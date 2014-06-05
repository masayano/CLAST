#include "krnlBinarySearch.cuh"

__global__ void binarySearch(
		const int limit,
		const int databaseSize,
		const long* databaseArray,
		const long* inputArray,
		int* outputArray) {
	const int bIdx = gridDim.x * blockIdx.y + blockIdx.x;
	const int tIdx = blockDim.x * bIdx + threadIdx.x;

	if(tIdx < limit) {
		const long input = inputArray[tIdx];
		int output = -1;

		int startIdx = 0;
		int endIdx = databaseSize - 1;
		while(startIdx <= endIdx) {
			const int idx = (startIdx + endIdx) >> 1;
			const long temp = databaseArray[idx];
			if(temp < input) {
				startIdx = idx + 1;
			} else if (temp == input) {
				output = idx;
				break;
			} else {
				endIdx = idx - 1;
			}
		}

		outputArray[tIdx] = output;
	}
}
