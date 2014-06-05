#include "krnlWriteSeedList.cuh"

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
		int* query_indexArray) {
	const int bIdx = blockIdx.y * gridDim.x + blockIdx.x;
	const int tIdx = blockDim.x * bIdx + threadIdx.x;
	if(tIdx < idxLimit) {
		const int qID  = qIDArray   [tIdx];
		const int qIdx = qIndexArray[tIdx];

		const int seedWriteIndex = seedWriteIndexArray[tIdx];
		const int cellSize       = cellSizeArray      [tIdx];

		const int gatewayIndex = gatewayIndexArray[tIdx];

		for(int i = 0; i < cellSize; ++i) {
			const int seedListIndex  = seedWriteIndex + i;
			const int refTargetIndex = indexArray[gatewayIndex + i];
			target_IDArray   [seedListIndex] = tIDArray   [refTargetIndex];
			target_indexArray[seedListIndex] = tIndexArray[refTargetIndex];
			query_IDArray    [seedListIndex] = qID;
			query_indexArray [seedListIndex] = qIdx;
		}
	}
}
