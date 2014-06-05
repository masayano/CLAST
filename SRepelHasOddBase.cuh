#ifndef S_REPEL_HAS_ODD_BASE_CUH_
#define S_REPEL_HAS_ODD_BASE_CUH_

#include <thrust/tuple.h>

struct repel_hasOddBase {
	__device__ long operator() (long index, char flg) {
		if(flg == 0) {
			return index;
		} else {
			return -1;
		}
	}
};

#endif
