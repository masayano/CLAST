#ifndef KRNL_BINARY_SEARCH_CUH_
#define KRNL_BINARY_SEARCH_CUH_

__global__ void binarySearch(
		const int limit,
		const int databaseSize,
		const long* databaseArray,
		const long* inputArray,
		int* outputArray);

#endif
