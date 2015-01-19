#include "krnlAlignment.cuh"
#include "krnlMatrix.cuh"

#include "common.hpp"

const int blockDim_x = 32;

__global__ void initTempNodeArray(
		const int hitNum,
		const int allowableGap,
		int* tempNodeArray_score,
		int* tempNodeArray_vertical,
		int* tempNodeArray_horizontal,
		int* tempNodeArray_matchNum) {
	const int bIdx = gridDim.x * blockIdx.y + blockIdx.x;
	const int idx  = blockDim.x * bIdx + threadIdx.x;
	const int halfTempNodeWidth = allowableGap + MARGIN;
	const int tempNodeWidth     = 1 + 2 * halfTempNodeWidth;
	if(idx < hitNum * tempNodeWidth) {
		const int bandIdx = idx / hitNum;
		if(bandIdx < halfTempNodeWidth) {
			tempNodeArray_score     [idx] = -30000;
			tempNodeArray_vertical  [idx] = -30000;
			tempNodeArray_horizontal[idx] = -30000;
			tempNodeArray_matchNum  [idx] = -30000;
		} else if(bandIdx == halfTempNodeWidth) {
			tempNodeArray_score     [idx] = 0;
			tempNodeArray_vertical  [idx] = GAP_OPEN_POINT;
			tempNodeArray_horizontal[idx] = GAP_OPEN_POINT;
			tempNodeArray_matchNum  [idx] = 0;
		} else {
			const int i = bandIdx - halfTempNodeWidth;
			const int tempScore = i * GAP_POINT + GAP_OPEN_POINT;
			tempNodeArray_score     [idx] = tempScore;
			tempNodeArray_vertical  [idx] = tempScore + GAP_OPEN_POINT;
			tempNodeArray_horizontal[idx] = tempScore;
			tempNodeArray_matchNum  [idx] = 0;
		}
	}
}

/**************************** one cell calculate function *****************************/

__device__ void calcutlateCell(
		const char targetBase,
		const char queryBase,
		const int verticalNode_score,
		const int verticalNode_vertical,
		const int verticalNode_matchNum,
		const int horizontalNode_score,
		const int horizontalNode_horizontal,
		const int horizontalNode_matchNum,
		int& slantNode_score,
		int& slantNode_vertical,
		int& slantNode_horizontal,
		int& slantNode_matchNum) {
	/* slantScore vertical */
	slantNode_vertical   = max(
			verticalNode_score    + GAP_OPEN_POINT + GAP_POINT,
			verticalNode_vertical + GAP_POINT);
	/* slantScore horizontal */
	slantNode_horizontal = max(
			horizontalNode_score      + GAP_OPEN_POINT + GAP_POINT,
			horizontalNode_horizontal + GAP_POINT);
	/* score */
	const int target = static_cast<int>(targetBase) - 65;
	const int query  = static_cast<int>(queryBase ) - 65;
	const int point  = slantNode_score + matrix::POINT_MATRIX[target][query];
	if(point > max(slantNode_vertical, slantNode_horizontal)) {
		slantNode_score     = point;
		slantNode_matchNum += matrix::MATCH_COUNTER[target][query];
	} else if(slantNode_vertical > slantNode_horizontal) {
		slantNode_score    = slantNode_vertical;
		slantNode_matchNum = verticalNode_matchNum;
	} else {
		slantNode_score    = slantNode_horizontal;
		slantNode_matchNum = horizontalNode_matchNum;
	}
}

/**************************** global alignment ********************************/

__global__ void globalAlignBackward(
		const int hitNum,
		const int allowableGap,
		const int t_begin,
		const int q_begin,
		const int*  tGateway,
		const int*  tLengthArray,
		const char* tBaseArray,
		const int*  qGateway,
		const int*  qLengthArray,
		const char* qBaseArray,
		const int* targetIDArray,
		const int* queryIDArray,
		const int* targetIndexArray,
		const int* queryIndexArray,
		int* tHitLengthArray,
		int* qHitLengthArray,
		int* matchNumArray,
		int* scoreArray,
		int* tempNodeArray_score,
		int* tempNodeArray_vertical,
		int* tempNodeArray_horizontal,
		int* tempNodeArray_matchNum) {
        const int bIdx = gridDim.x * blockIdx.y + blockIdx.x;
	const int hitIdx = blockDim.x * bIdx + threadIdx.x;
	if(hitIdx < hitNum) {
		/* define of consts (common) */
		const int tID        = targetIDArray   [hitIdx];
		const int qID        = queryIDArray    [hitIdx];
		const int tIdx       = targetIndexArray[hitIdx];
		const int qIdx       = queryIndexArray [hitIdx];
		const int tStartIdx  = tGateway        [tID - t_begin];
		const int qStartIdx  = qGateway        [qID - q_begin];
		const int tLength    = tLengthArray    [tID - t_begin];
		const int qLength    = qLengthArray    [qID - q_begin];
		const int qHitLength = qHitLengthArray[hitIdx];

		/* initialize target base */
		__shared__ char targetBaseArray[blockDim_x][MAX_ALIGNMENT_WIDTH - 2];
		char* pTargetBaseArray = targetBaseArray[threadIdx.x];
		const int alignmentWidth     = 1 + 2 * allowableGap;
		for(int t = 0; t < alignmentWidth; ++t) {
			const int tBaseIdx = tIdx + qHitLength - allowableGap + t;
			if(tBaseIdx >= tLength) {
				pTargetBaseArray[t] = BAD_AREA_CHAR;
			} else {
				pTargetBaseArray[t] = tBaseArray[tStartIdx + tBaseIdx];
			}
		}

		/* alignment */
		int shrdTargetIdx = 0;
		const int alignmentStartIdx = qIdx + qHitLength;
		const int alignmentSize     = qLength - alignmentStartIdx;
		for(int q = 0; q < alignmentSize; ++q) {
			const char queryBase = qBaseArray[qStartIdx + alignmentStartIdx + q];
			int* pTempNode_score      = tempNodeArray_score      + hitIdx;
			int* pTempNode_vertical   = tempNodeArray_vertical   + hitIdx;
			int* pTempNode_horizontal = tempNodeArray_horizontal + hitIdx;
			int* pTempNode_matchNum   = tempNodeArray_matchNum   + hitIdx;
			for(int t = 0; t < alignmentWidth; ++t) {
				pTempNode_score      += hitNum;
				pTempNode_vertical   += hitNum;
				pTempNode_horizontal += hitNum;
				pTempNode_matchNum   += hitNum;
				calcutlateCell(
						pTargetBaseArray[shrdTargetIdx++],
						queryBase,
						*(pTempNode_score    + hitNum),//
						*(pTempNode_vertical + hitNum),// lower
						*(pTempNode_matchNum + hitNum),//
						*(pTempNode_score      - hitNum),//
						*(pTempNode_horizontal - hitNum),// left
						*(pTempNode_matchNum   - hitNum),//
						*pTempNode_score,     //
						*pTempNode_vertical,  // slant
						*pTempNode_horizontal,//
						*pTempNode_matchNum); //
				if(shrdTargetIdx == alignmentWidth) { shrdTargetIdx = 0; }
			}
			const int tBaseIdx = tIdx + qHitLength + allowableGap + q + 1;
			if(tBaseIdx >= tLength) {
				pTargetBaseArray[shrdTargetIdx] = BAD_AREA_CHAR;
			} else {
				pTargetBaseArray[shrdTargetIdx] = tBaseArray[tStartIdx + tBaseIdx];
			}
			++shrdTargetIdx;
			if(shrdTargetIdx == alignmentWidth) { shrdTargetIdx = 0; }
		}

		/* select end node (common) */
		int endPoint = -1;
		int* pTempNode_score    = tempNodeArray_score    + hitIdx;
		int* pTempNode_matchNum = tempNodeArray_matchNum + hitIdx;
		int scoreNode_score    = *pTempNode_score;
		int scoreNode_matchNum = *pTempNode_matchNum;
		for(int t = 0; t < alignmentWidth; ++t) {
			pTempNode_score    += hitNum;
			pTempNode_matchNum += hitNum;
			if(scoreNode_score < *pTempNode_score) {
				scoreNode_score    = *pTempNode_score;
				scoreNode_matchNum = *pTempNode_matchNum;
				endPoint = t;
			}
		}
		endPoint -= allowableGap;

		/* write result */
		if(alignmentSize != 0) {
			tHitLengthArray[hitIdx] += alignmentSize + endPoint;
			qHitLengthArray[hitIdx] += alignmentSize;
			matchNumArray  [hitIdx] += scoreNode_matchNum;
			scoreArray     [hitIdx] += scoreNode_score;
		}
	}
}

__global__ void globalAlignForward(
		const int hitNum,
		const int allowableGap,
		const int t_begin,
		const int q_begin,
		const int*  tGateway,
		const int*  tLengthArray,
		const char* tBaseArray,
		const int*  qGateway,
		const int*  qLengthArray,
		const char* qBaseArray,
		const int* targetIDArray,
		const int* queryIDArray,
		int* targetIndexArray,
		int* queryIndexArray,
		int* tHitLengthArray,
		int* qHitLengthArray,
		int* matchNumArray,
		int* scoreArray,
		int* tempNodeArray_score,
		int* tempNodeArray_vertical,
		int* tempNodeArray_horizontal,
		int* tempNodeArray_matchNum) {
        const int bIdx = gridDim.x * blockIdx.y + blockIdx.x;
        const int hitIdx = blockDim.x * bIdx + threadIdx.x;
	if(hitIdx < hitNum) {
		/* define of consts (common) */
		const int tID  = targetIDArray   [hitIdx];
		const int qID  = queryIDArray    [hitIdx];
		const int tIdx = targetIndexArray[hitIdx];
		const int qIdx = queryIndexArray [hitIdx];
		const int tStartIdx = tGateway[tID - t_begin];
		const int qStartIdx = qGateway[qID - q_begin];

		/* initialize target base */
		__shared__ char targetBaseArray[blockDim_x][MAX_ALIGNMENT_WIDTH - 2];
		char* pTargetBaseArray = targetBaseArray[threadIdx.x];
		const int alignmentWidth = 1 + 2 * allowableGap;
		for(int t = 0; t < alignmentWidth; ++t) {
			const int tBaseIdx = tIdx + allowableGap - t - 1;
			if(tBaseIdx < 0) {
				pTargetBaseArray[t] = BAD_AREA_CHAR;
			} else {
				pTargetBaseArray[t] = tBaseArray[tStartIdx + tBaseIdx];
			}
		}

		/* alignment */
		int shrdTargetIdx = 0;
		for(int q = 0; q < qIdx; ++q) {
			const char queryBase = qBaseArray[qStartIdx + qIdx - q - 1];
			int* pTempNode_score      = tempNodeArray_score      + hitIdx;
			int* pTempNode_vertical   = tempNodeArray_vertical   + hitIdx;
			int* pTempNode_horizontal = tempNodeArray_horizontal + hitIdx;
			int* pTempNode_matchNum   = tempNodeArray_matchNum   + hitIdx;
			for(int t = 0; t < alignmentWidth; ++t) {
				pTempNode_score      += hitNum;
				pTempNode_vertical   += hitNum;
				pTempNode_horizontal += hitNum;
				pTempNode_matchNum   += hitNum;
				calcutlateCell(
						pTargetBaseArray[shrdTargetIdx++],
						queryBase,
						*(pTempNode_score    + hitNum),//
						*(pTempNode_vertical + hitNum),// upper
						*(pTempNode_matchNum + hitNum),//
						*(pTempNode_score      - hitNum),//
						*(pTempNode_horizontal - hitNum),// right
						*(pTempNode_matchNum   - hitNum),//
						*pTempNode_score,     //
						*pTempNode_vertical,  // slant
						*pTempNode_horizontal,//
						*pTempNode_matchNum); //
				if(shrdTargetIdx == alignmentWidth) { shrdTargetIdx = 0; }
			}
			const int tBaseIdx = tIdx - allowableGap - q - 2;
			if(tBaseIdx < 0) {
				pTargetBaseArray[shrdTargetIdx] = BAD_AREA_CHAR;
			} else {
				pTargetBaseArray[shrdTargetIdx] = tBaseArray[tStartIdx + tBaseIdx];
			}
			++shrdTargetIdx;
			if(shrdTargetIdx == alignmentWidth) { shrdTargetIdx = 0; }
		}

		/* select end node (common) */
		int endPoint = -1;
		int* pTempNode_score    = tempNodeArray_score    + hitIdx;
		int* pTempNode_matchNum = tempNodeArray_matchNum + hitIdx;
		int scoreNode_score    = *pTempNode_score;
		int scoreNode_matchNum = *pTempNode_matchNum;
		for(int t = 0; t < alignmentWidth; ++t) {
			pTempNode_score    += hitNum;
			pTempNode_matchNum += hitNum;
			if(scoreNode_score < *pTempNode_score) {
				scoreNode_score    = *pTempNode_score;
				scoreNode_matchNum = *pTempNode_matchNum;
				endPoint = t;
			}
		}
		endPoint -= allowableGap;

		/* write result */
		if(qIdx != 0) {
			targetIndexArray[hitIdx] -= qIdx + endPoint;
			queryIndexArray [hitIdx] -= qIdx;
			tHitLengthArray [hitIdx] += qIdx + endPoint;
			qHitLengthArray [hitIdx] += qIdx;
			matchNumArray   [hitIdx] += scoreNode_matchNum;
			scoreArray      [hitIdx] += scoreNode_score;
		}
	}
}

/**************************** local alignment ********************************/

__global__ void localAlignBackward(
		const int hitNum,
		const int allowableGap,
		const int t_begin,
		const int q_begin,
		const int*  tGateway,
		const int*  tLengthArray,
		const char* tBaseArray,
		const int*  qGateway,
		const int*  qLengthArray,
		const char* qBaseArray,
		const int* targetIDArray,
		const int* queryIDArray,
		const int* targetIndexArray,
		const int* queryIndexArray,
		int* tHitLengthArray,
		int* qHitLengthArray,
		int* matchNumArray,
		int* scoreArray,
		int* tempNodeArray_score,
		int* tempNodeArray_vertical,
		int* tempNodeArray_horizontal,
		int* tempNodeArray_matchNum) {
        const int bIdx = gridDim.x * blockIdx.y + blockIdx.x;
        const int hitIdx = blockDim.x * bIdx + threadIdx.x;
	if(hitIdx < hitNum) {
		/* define of consts (common) */
		const int tID       = targetIDArray   [hitIdx];
		const int qID       = queryIDArray    [hitIdx];
		const int tIdx      = targetIndexArray[hitIdx];
		const int qIdx      = queryIndexArray [hitIdx];
		const int tStartIdx = tGateway        [tID - t_begin];
		const int qStartIdx = qGateway        [qID - q_begin];
		const int tLength   = tLengthArray    [tID - t_begin];
		const int qLength   = qLengthArray    [qID - q_begin];
		const int qHitLength = qHitLengthArray[hitIdx];

		/* initialize target base */
		__shared__ char targetBaseArray[blockDim_x][MAX_ALIGNMENT_WIDTH - 2];
		char* pTargetBaseArray = targetBaseArray[threadIdx.x];
		const int alignmentWidth = 1 + 2 * allowableGap;
		for(int t = 0; t < alignmentWidth; ++t) {
			const int tBaseIdx = tIdx + qHitLength - allowableGap + t;
			if(tBaseIdx >= tLength) {
				pTargetBaseArray[t] = BAD_AREA_CHAR;
			} else {
				pTargetBaseArray[t] = tBaseArray[tStartIdx + tBaseIdx];
			}
		}

		/* prepare end node */
		int calcIdx  = -1;
		int endPoint = allowableGap;
		int scoreNode_score    = 0;
		int scoreNode_matchNum = 0;
		/* alignment */
		int shrdTargetIdx = 0;
		const int alignmentStartIdx = qIdx + qHitLength;
		const int alignmentSize     = qLength - alignmentStartIdx;
		for(int q = 0; q < alignmentSize; ++q) {
			const char queryBase = qBaseArray[qStartIdx + alignmentStartIdx + q];
			int* pTempNode_score      = tempNodeArray_score      + hitIdx;
			int* pTempNode_vertical   = tempNodeArray_vertical   + hitIdx;
			int* pTempNode_horizontal = tempNodeArray_horizontal + hitIdx;
			int* pTempNode_matchNum   = tempNodeArray_matchNum   + hitIdx;
			for(int t = 0; t < alignmentWidth; ++t) {
				pTempNode_score      += hitNum;
				pTempNode_vertical   += hitNum;
				pTempNode_horizontal += hitNum;
				pTempNode_matchNum   += hitNum;
				calcutlateCell(
						pTargetBaseArray[shrdTargetIdx++],
						queryBase,
						*(pTempNode_score    + hitNum),//
						*(pTempNode_vertical + hitNum),// lower
						*(pTempNode_matchNum + hitNum),//
						*(pTempNode_score      - hitNum),//
						*(pTempNode_horizontal - hitNum),// left
						*(pTempNode_matchNum   - hitNum),//
						*pTempNode_score,     //
						*pTempNode_vertical,  // slant
						*pTempNode_horizontal,//
						*pTempNode_matchNum); //
				if(shrdTargetIdx == alignmentWidth) { shrdTargetIdx = 0; }
				if(scoreNode_score < *pTempNode_score) {
					scoreNode_score    = *pTempNode_score;
					scoreNode_matchNum = *pTempNode_matchNum;
					calcIdx   = q;
					endPoint  = t;
				}
			}
			const int tBaseIdx = tIdx + qHitLength + allowableGap + q + 1;
			if(tBaseIdx >= tLength) {
				pTargetBaseArray[shrdTargetIdx] = BAD_AREA_CHAR;
			} else {
				pTargetBaseArray[shrdTargetIdx] = tBaseArray[tStartIdx + tBaseIdx];
			}
			++shrdTargetIdx;
			if(shrdTargetIdx == alignmentWidth) { shrdTargetIdx = 0; }
		}
		/* preprocess end point */
		endPoint -= allowableGap;

		/* write result */
		if(alignmentSize != 0) {
			const int moveLength = calcIdx + 1;
			tHitLengthArray[hitIdx] += moveLength + endPoint;
			qHitLengthArray[hitIdx] += moveLength;
			matchNumArray  [hitIdx] += scoreNode_matchNum;
			scoreArray     [hitIdx] += scoreNode_score;
		}
	}
}

__global__ void localAlignForward(
		const int hitNum,
		const int allowableGap,
		const int t_begin,
		const int q_begin,
		const int*  tGateway,
		const int*  tLengthArray,
		const char* tBaseArray,
		const int*  qGateway,
		const int*  qLengthArray,
		const char* qBaseArray,
		const int* targetIDArray,
		const int* queryIDArray,
		int* targetIndexArray,
		int* queryIndexArray,
		int* tHitLengthArray,
		int* qHitLengthArray,
		int* matchNumArray,
		int* scoreArray,
		int* tempNodeArray_score,
		int* tempNodeArray_vertical,
		int* tempNodeArray_horizontal,
		int* tempNodeArray_matchNum) {
        const int bIdx = gridDim.x * blockIdx.y + blockIdx.x;
        const int hitIdx = blockDim.x * bIdx + threadIdx.x;
	if(hitIdx < hitNum) {
		/* define of consts (common) */
		const int tID  = targetIDArray   [hitIdx];
		const int qID  = queryIDArray    [hitIdx];
		const int tIdx = targetIndexArray[hitIdx];
		const int qIdx = queryIndexArray [hitIdx];
		const int tStartIdx = tGateway[tID - t_begin];
		const int qStartIdx = qGateway[qID - q_begin];

		/* initialize target base on resister */
		__shared__ char targetBaseArray[blockDim_x][MAX_ALIGNMENT_WIDTH - 2];
		char* pTargetBaseArray = targetBaseArray[threadIdx.x];
		const int alignmentWidth = 1 + 2 * allowableGap;
		for(int t = 0; t < alignmentWidth; ++t) {
			const int tBaseIdx = tIdx + allowableGap - t - 1;
			if(tBaseIdx < 0) {
				pTargetBaseArray[t] = BAD_AREA_CHAR;
			} else {
				pTargetBaseArray[t] = tBaseArray[tStartIdx + tBaseIdx];
			}
		}

		/* prepare end node */
		int calcIdx  = -1;
		int endPoint = allowableGap;
		int scoreNode_score    = 0;
		int scoreNode_matchNum = 0;
		/* alignment */
		int shrdTargetIdx = 0;
		for(int q = 0; q < qIdx; ++q) {
			const char queryBase = qBaseArray[qStartIdx + qIdx - q - 1];
			int* pTempNode_score      = tempNodeArray_score      + hitIdx;
			int* pTempNode_vertical   = tempNodeArray_vertical   + hitIdx;
			int* pTempNode_horizontal = tempNodeArray_horizontal + hitIdx;
			int* pTempNode_matchNum   = tempNodeArray_matchNum   + hitIdx;
			for(int t = 0; t < alignmentWidth; ++t) {
				pTempNode_score      += hitNum;
				pTempNode_vertical   += hitNum;
				pTempNode_horizontal += hitNum;
				pTempNode_matchNum   += hitNum;
				calcutlateCell(
						pTargetBaseArray[shrdTargetIdx++],
						queryBase,
						*(pTempNode_score    + hitNum),//
						*(pTempNode_vertical + hitNum),// upper
						*(pTempNode_matchNum + hitNum),//
						*(pTempNode_score      - hitNum),//
						*(pTempNode_horizontal - hitNum),// right
						*(pTempNode_matchNum   - hitNum),//
						*pTempNode_score,     //
						*pTempNode_vertical,  // slant
						*pTempNode_horizontal,//
						*pTempNode_matchNum); //
				if(shrdTargetIdx == alignmentWidth) { shrdTargetIdx = 0; }
				if(scoreNode_score < *pTempNode_score) {
					scoreNode_score    = *pTempNode_score;
					scoreNode_matchNum = *pTempNode_matchNum;
					calcIdx   = q;
					endPoint  = t;
				}
			}
			const int tBaseIdx = tIdx - allowableGap - q - 2;
			if(tBaseIdx < 0) {
				pTargetBaseArray[shrdTargetIdx] = BAD_AREA_CHAR;
			} else {
				pTargetBaseArray[shrdTargetIdx] = tBaseArray[tStartIdx + tBaseIdx];
			}
			++shrdTargetIdx;
			if(shrdTargetIdx == alignmentWidth) { shrdTargetIdx = 0; }
		}
		/* preprocess end point */
		endPoint -= allowableGap;

		/* write result */
		if(qIdx != 0) {
			const int moveLength = calcIdx + 1;
			targetIndexArray[hitIdx] -= moveLength + endPoint;
			queryIndexArray [hitIdx] -= moveLength;
			tHitLengthArray [hitIdx] += moveLength + endPoint;
			qHitLengthArray [hitIdx] += moveLength;
			matchNumArray   [hitIdx] += scoreNode_matchNum;
			scoreArray      [hitIdx] += scoreNode_score;
		}
	}
}
