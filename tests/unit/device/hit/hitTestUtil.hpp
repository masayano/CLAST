// Shared Thrust host-vector helpers for tests under this directory. Each
// .cpp is titled after the `src/device/hit/*.cu` unit under test.
#pragma once

#include <initializer_list>
#include <thrust/host_vector.h>

namespace clast {
namespace test {
namespace hit {

inline thrust::host_vector<int> MakeIntVec(std::initializer_list<int> xs) {
	thrust::host_vector<int> v;
	v.reserve(xs.size());
	for (int x : xs) {
		v.push_back(x);
	}
	return v;
}

inline thrust::host_vector<double> MakeDoubleVec(std::initializer_list<double> xs) {
	thrust::host_vector<double> v;
	v.reserve(xs.size());
	for (double x : xs) {
		v.push_back(x);
	}
	return v;
}

inline thrust::host_vector<long> MakeLongVec(std::initializer_list<long> xs) {
	thrust::host_vector<long> v;
	v.reserve(xs.size());
	for (long x : xs) {
		v.push_back(x);
	}
	return v;
}


}  // namespace hit
}  // namespace test
}  // namespace clast
