// Shared Thrust host-vector helpers for tests under this directory. Each
// .cpp is titled after the `src/device/hit/*.cu` unit under test.
#pragma once

#include <initializer_list>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>

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

inline thrust::tuple<int, int, int, int> RowAt(
		const thrust::host_vector<int>& tID,
		const thrust::host_vector<int>& tIdx,
		const thrust::host_vector<int>& qID,
		const thrust::host_vector<int>& qIdx,
		int i) {
	return thrust::make_tuple(tID[i], tIdx[i], qID[i], qIdx[i]);
}

}  // namespace hit
}  // namespace test
}  // namespace clast
