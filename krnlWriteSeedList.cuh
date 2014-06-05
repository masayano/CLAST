#ifndef KRNL_WRITE_HIT_LIST_CUH_
#define KRNL_WRITE_HIT_LIST_CUH_

__global__ void writeSeedList(
		const int idxLimit,
		const int* gatewayIndexArray,
		const int* indexArray,
		const int* seedWriteIndexArray,
		const int* cellSizeArray,
		const int* tIDArray,
		const int* tIndexArray,
		const int* qIDArray,
		const int* qIndexArray,
		int* target_IDArray,
		int* target_indexArray,
		int* query_IDArray,
		int* query_indexArray);

#endif
