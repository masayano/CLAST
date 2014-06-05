#ifndef KRNL_CALCULATE_EVALUE_CUH_
#define KRNL_CALCULATE_EVALUE_CUH_

__global__ void calculateEvalue(
		const int q_begin,
		const int matchSize,
		const double totalDatabaseSize,
		const double K,
		const double lambda,
		const int* queryLengthArray,
		const int* queryIDArray,
		const int* scoreArray,
		double* evalueArray);

#endif /* KRNL_CALCULATE_EVALUE_CUH_ */
