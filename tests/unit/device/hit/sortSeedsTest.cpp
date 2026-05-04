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

// Input:   (tID,tIdx,qID,qIdx) = (0,10,1,0), (0,9,0,5), (1,3,0,4)
// Sort by qID → tID → dia(tIdx-qIdx) → qIdx:
//   qID=0: tID=0→(0,9,0,5), tID=1→(1,3,0,4)  tID 0 < 1 → (0,9,0,5) first
//   qID=1: (0,10,1,0)
// Expected row order: (0,9,0,5), (1,3,0,4), (0,10,1,0)
TEST(SortSeeds, OrdersByQueryThenTargetThenDiagonalThenQueryIndex) {
	auto tID  = MakeIntVec({0,  0, 1});
	auto tIdx = MakeIntVec({10, 9, 3});
	auto qID  = MakeIntVec({1,  0, 0});
	auto qIdx = MakeIntVec({0,  5, 4});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(tID[0],  0); EXPECT_EQ(tID[1],  1); EXPECT_EQ(tID[2],  0);
	EXPECT_EQ(tIdx[0], 9); EXPECT_EQ(tIdx[1], 3); EXPECT_EQ(tIdx[2], 10);
	EXPECT_EQ(qID[0],  0); EXPECT_EQ(qID[1],  0); EXPECT_EQ(qID[2],  1);
	EXPECT_EQ(qIdx[0], 5); EXPECT_EQ(qIdx[1], 4); EXPECT_EQ(qIdx[2],  0);
}

// Primary key: qID. tID/tIdx/qIdx identical between rows; only qID differs.
// Input:  (0,10,5,3), (0,10,2,3)
// Expected: (0,10,2,3), (0,10,5,3)
TEST(SortSeeds, TiebreaksByQueryID) {
	auto tID  = MakeIntVec({0,  0});
	auto tIdx = MakeIntVec({10, 10});
	auto qID  = MakeIntVec({5,  2});
	auto qIdx = MakeIntVec({3,  3});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(tID[0],  0); EXPECT_EQ(tID[1],  0);
	EXPECT_EQ(tIdx[0], 10); EXPECT_EQ(tIdx[1], 10);
	EXPECT_EQ(qID[0],  2); EXPECT_EQ(qID[1],  5);
	EXPECT_EQ(qIdx[0], 3); EXPECT_EQ(qIdx[1], 3);
}

// Secondary key: tID. qID/tIdx/qIdx identical; only tID differs.
// Input:  (7,10,1,4), (3,10,1,4)
// Expected: (3,10,1,4), (7,10,1,4)
TEST(SortSeeds, TiebreaksByTargetIDWhenQueryIDEqual) {
	auto tID  = MakeIntVec({7,  3});
	auto tIdx = MakeIntVec({10, 10});
	auto qID  = MakeIntVec({1,  1});
	auto qIdx = MakeIntVec({4,  4});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(tID[0],  3); EXPECT_EQ(tID[1],  7);
	EXPECT_EQ(tIdx[0], 10); EXPECT_EQ(tIdx[1], 10);
	EXPECT_EQ(qID[0],  1); EXPECT_EQ(qID[1],  1);
	EXPECT_EQ(qIdx[0], 4); EXPECT_EQ(qIdx[1], 4);
}

// Tertiary key: diagonal = tIdx - qIdx. qID and tID equal; diagonal differs.
// Seed A: (0,20,0,5)  dia=15
// Seed B: (0,30,0,20) dia=10   → B first (10 < 15)
// Expected: (0,30,0,20), (0,20,0,5)
TEST(SortSeeds, TiebreaksByDiagonalWhenQueryAndTargetIDEqual) {
	auto tID  = MakeIntVec({0,  0});
	auto tIdx = MakeIntVec({20, 30});
	auto qID  = MakeIntVec({0,  0});
	auto qIdx = MakeIntVec({5,  20});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(tID[0],  0); EXPECT_EQ(tID[1],  0);
	EXPECT_EQ(tIdx[0], 30); EXPECT_EQ(tIdx[1], 20);
	EXPECT_EQ(qID[0],  0); EXPECT_EQ(qID[1],  0);
	EXPECT_EQ(qIdx[0], 20); EXPECT_EQ(qIdx[1], 5);
}

// Quaternary key: qIdx. qID, tID, diagonal all equal; only qIdx (and tIdx) differ.
// Seed A: (0,15,0,5)  dia=10
// Seed B: (0,12,0,2)  dia=10   → B first (qIdx 2 < 5)
// Expected: (0,12,0,2), (0,15,0,5)
TEST(SortSeeds, TiebreaksByQueryIndexWhenDiagonalEqual) {
	auto tID  = MakeIntVec({0,  0});
	auto tIdx = MakeIntVec({15, 12});
	auto qID  = MakeIntVec({0,  0});
	auto qIdx = MakeIntVec({5,  2});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(tID[0],  0); EXPECT_EQ(tID[1],  0);
	EXPECT_EQ(tIdx[0], 12); EXPECT_EQ(tIdx[1], 15);
	EXPECT_EQ(qID[0],  0); EXPECT_EQ(qID[1],  0);
	EXPECT_EQ(qIdx[0], 2); EXPECT_EQ(qIdx[1], 5);
}

// All four keys exercised together with fully reversed input.
// Input rows (tID,tIdx,qID,qIdx): (1,10,1,4), (0,20,0,5 dia=15), (0,15,0,5 dia=10), (0,12,0,2 dia=10)
// Sort: qID=0 before qID=1; within qID=0 all tID=0; dia 10 < 15 → rows 2,3 before row 1;
//       rows 2,3 have same dia, qIdx 2 < 5 → row 3 before row 2.
// Expected order: (0,12,0,2), (0,15,0,5), (0,20,0,5), (1,10,1,4)
TEST(SortSeeds, ReversedInputGetsSorted) {
	auto tID  = MakeIntVec({1,  0,  0,  0});
	auto tIdx = MakeIntVec({10, 20, 15, 12});
	auto qID  = MakeIntVec({1,  0,  0,  0});
	auto qIdx = MakeIntVec({4,  5,  5,  2});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(tID[0],  0); EXPECT_EQ(tID[1],  0); EXPECT_EQ(tID[2],  0); EXPECT_EQ(tID[3],  1);
	EXPECT_EQ(tIdx[0], 12); EXPECT_EQ(tIdx[1], 15); EXPECT_EQ(tIdx[2], 20); EXPECT_EQ(tIdx[3], 10);
	EXPECT_EQ(qID[0],  0); EXPECT_EQ(qID[1],  0); EXPECT_EQ(qID[2],  0); EXPECT_EQ(qID[3],  1);
	EXPECT_EQ(qIdx[0], 2); EXPECT_EQ(qIdx[1],  5); EXPECT_EQ(qIdx[2],  5); EXPECT_EQ(qIdx[3], 4);
}
