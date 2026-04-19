#ifndef S_READ_2_BIT_CUH_
#define S_READ_2_BIT_CUH_

#include <thrust/tuple.h>

struct read2bit {
	template <typename Tuple>
	__device__ Tuple operator() (const char chr, Tuple tuple) {
		long input = thrust::get<0>(tuple);
		char flg   = thrust::get<1>(tuple);
		input <<= 2;
		switch(chr) {
			case 'A': input += 0; break;
			case 'C': input += 1; break;
			case 'G': input += 2; break;
			case 'T': input += 3; break;
			default : flg = 1;
		}
		return thrust::tie(input, flg);
	}
};

#endif
