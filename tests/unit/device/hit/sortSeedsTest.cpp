// Unit tests for `src/device/hit/sortSeeds.cu` / `sortSeeds.cuh` (host Thrust
// `measureDistanceSorting` and `measure_distance`).

#include "device/hit/sortSeeds.cuh"
#include "hitSortSeedsExpect.hpp"

#include <gtest/gtest.h>

using clast::test::hit::ExpectSortedByMeasureDistance;
using clast::test::hit::MakeIntVec;

TEST(SortSeeds, EmptyUnchanged) {
	auto tID = MakeIntVec({});
	auto tIdx = MakeIntVec({});
	auto qID = MakeIntVec({});
	auto qIdx = MakeIntVec({});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

TEST(SortSeeds, SingleRowUnchanged) {
	auto tID = MakeIntVec({1});
	auto tIdx = MakeIntVec({2});
	auto qID = MakeIntVec({3});
	auto qIdx = MakeIntVec({4});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(tID[0], 1);
	EXPECT_EQ(tIdx[0], 2);
	EXPECT_EQ(qID[0], 3);
	EXPECT_EQ(qIdx[0], 4);
}

TEST(SortSeeds, OrdersByQueryThenTargetThenDiagonalThenQueryIndex) {
	auto tID = MakeIntVec({0, 0, 1});
	auto tIdx = MakeIntVec({10, 9, 3});
	auto qID = MakeIntVec({1, 0, 0});
	auto qIdx = MakeIntVec({0, 5, 4});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(qID[0], 0);
	EXPECT_EQ(qID[1], 0);
	EXPECT_EQ(qID[2], 1);
}
