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

// Primary key: qID. Two seeds with same tID/tIdx/qIdx but different qID.
TEST(SortSeeds, TiebreaksByQueryID) {
	auto tID  = MakeIntVec({0,  0});
	auto tIdx = MakeIntVec({10, 10});
	auto qID  = MakeIntVec({5,  2});  // 2 < 5 → row with qID=2 must come first
	auto qIdx = MakeIntVec({3,  3});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(qID[0], 2);
	EXPECT_EQ(qID[1], 5);
}

// Secondary key: tID (qID equal, tID differs).
TEST(SortSeeds, TiebreaksByTargetIDWhenQueryIDEqual) {
	auto tID  = MakeIntVec({7,  3});  // 3 < 7 → row with tID=3 must come first
	auto tIdx = MakeIntVec({10, 10});
	auto qID  = MakeIntVec({1,  1});
	auto qIdx = MakeIntVec({4,  4});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(tID[0], 3);
	EXPECT_EQ(tID[1], 7);
}

// Tertiary key: diagonal = tIdx - qIdx (qID and tID equal, diagonal differs).
// Seed A: tIdx=20, qIdx=5  → dia=15
// Seed B: tIdx=30, qIdx=20 → dia=10   (10 < 15 → B first)
TEST(SortSeeds, TiebreaksByDiagonalWhenQueryAndTargetIDEqual) {
	auto tID  = MakeIntVec({0,  0});
	auto tIdx = MakeIntVec({20, 30});
	auto qID  = MakeIntVec({0,  0});
	auto qIdx = MakeIntVec({5,  20});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	// diagonal 10 < 15 → the seed with tIdx=30,qIdx=20 should be first
	EXPECT_EQ(tIdx[0], 30);
	EXPECT_EQ(qIdx[0], 20);
	EXPECT_EQ(tIdx[1], 20);
	EXPECT_EQ(qIdx[1], 5);
}

// Quaternary key: qIdx (qID, tID, diagonal all equal).
// Same diagonal means tIdx - qIdx is constant; only qIdx (and tIdx) differ.
// Seed A: tIdx=15, qIdx=5  → dia=10
// Seed B: tIdx=12, qIdx=2  → dia=10   qIdx 2 < 5 → B first
TEST(SortSeeds, TiebreaksByQueryIndexWhenDiagonalEqual) {
	auto tID  = MakeIntVec({0,  0});
	auto tIdx = MakeIntVec({15, 12});
	auto qID  = MakeIntVec({0,  0});
	auto qIdx = MakeIntVec({5,  2});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(qIdx[0], 2);
	EXPECT_EQ(qIdx[1], 5);
}

// Fully reversed input is correctly sorted (exercises all key levels together).
TEST(SortSeeds, ReversedInputGetsSorted) {
	// Descending by (qID, tID, dia, qIdx) → ascending output
	auto tID  = MakeIntVec({1,  0,  0,  0});
	auto tIdx = MakeIntVec({10, 20, 15, 12});
	auto qID  = MakeIntVec({1,  0,  0,  0});
	auto qIdx = MakeIntVec({4,  5,  5,  2});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	// First element must have smallest qID
	EXPECT_EQ(qID[0], 0);
	// Last element must have largest qID
	EXPECT_EQ(qID[3], 1);
}
