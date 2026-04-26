// Assertions for `measureDistanceSorting` (declared in sortSeeds.cuh).
#pragma once

#include "device/hit/sortSeeds.cuh"
#include "hitTestUtil.hpp"

#include <gtest/gtest.h>

namespace clast {
namespace test {
namespace hit {

inline void ExpectSortedByMeasureDistance(
		const thrust::host_vector<int>& tID,
		const thrust::host_vector<int>& tIdx,
		const thrust::host_vector<int>& qID,
		const thrust::host_vector<int>& qIdx) {
	ASSERT_EQ(tID.size(), tIdx.size());
	ASSERT_EQ(tID.size(), qID.size());
	ASSERT_EQ(tID.size(), qIdx.size());
	const int n = static_cast<int>(tID.size());
	measure_distance cmp;
	for (int i = 0; i < n - 1; ++i) {
		const auto a = RowAt(tID, tIdx, qID, qIdx, i);
		const auto b = RowAt(tID, tIdx, qID, qIdx, i + 1);
		EXPECT_FALSE(cmp(b, a))
				<< "rows " << i << " and " << (i + 1) << " are out of order";
	}
}

}  // namespace hit
}  // namespace test
}  // namespace clast
