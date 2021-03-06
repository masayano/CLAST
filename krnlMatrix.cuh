#ifndef KRNL_MATRIX_CUH_
#define KRNL_MATRIX_CUH_

#include "common.hpp"

namespace matrix {
	const int o = MATCH_POINT;
	const int x = MISMATCH_POINT;
	const int b = BAD_AREA_POINT;

	__device__ __constant__ int POINT_MATRIX[26][26] = {
	    // A   B   C   D   *   *   G   H   *   *   K   *   M   N   *   *   *   R   S   T   U   V   W   *   Y   *
	/*A*/{ o,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/*B*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/*C*/{ x,  x,  o,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/*D*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},              
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/*G*/{ x,  x,  x,  x, -1, -1,  o,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},             
	/*H*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/*K*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/*M*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/*N*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/*R*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/*S*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/*T*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  o,  x,  x,  x,  b,  x, -1},
	/*U*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/*V*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/*W*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/***/{ b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b},
	/*Y*/{ x,  x,  x,  x, -1, -1,  x,  x, -1, -1,  x, -1,  x,  x, -1, -1, -1,  x,  x,  x,  x,  x,  x,  b,  x, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1}};

	__device__ __constant__ int MATCH_COUNTER[26][26] = { 
	    // A   B   C   D   *   *   G   H   *   *   K   *   M   N   *   *   *   R   S   T   U   V   W   *   Y   *
	/*A*/{ 1,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/*B*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/*C*/{ 0,  0,  1,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/*D*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},            
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/*G*/{ 0,  0,  0,  0, -1, -1,  1,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/*H*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/*K*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/*M*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/*N*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1},
	/*R*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/*S*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/*T*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  1,  0,  0,  0,  b,  0, -1},
	/*U*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/*V*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/*W*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/***/{ b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b,  b},
	/*Y*/{ 0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0, -1,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  b,  0, -1},
	/***/{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  b, -1, -1}};
}

#endif
