#ifndef KRNL_ALIGNMENT_CUH_
#define KRNL_ALIGNMENT_CUH_

__global__ void initTempNodeArray(
		const int hitNum,
		const int allowableGap,
		int* tempNodeArray_score,
		int* tempNodeArray_vertical,
		int* tempNodeArray_horizontal,
		int* tempNodeArray_matchNum);

/*********************** global alignment ***********************/

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
		int* tempNodeArray_matchNum);

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
		int* tempNodeArray_matchNum);

/*********************** local alignment ***********************/

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
		int* tempNodeArray_matchNum);

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
		int* tempNodeArray_matchNum);

#endif /*KRNL_ALLIGNMENT_CUH_  */
