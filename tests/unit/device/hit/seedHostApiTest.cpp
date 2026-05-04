// Unit tests for host-side Thrust seed helpers (`seedHostApi.cuh` → `clast::hit::*` in
// `seedThrustVectorOps.cuh`): duplicate removal, isolate removal, boundary filters, corner filter.
// Inputs use `measureDistanceSorting` from `sortSeeds.cuh` when sort order matters.

#include "device/hit/seedHostApi.cuh"
#include "device/hit/sortSeeds.cuh"
#include "hitTestUtil.hpp"

#include <gtest/gtest.h>

using clast::test::hit::MakeIntVec;

// `deleteDuplicateSeeds` only runs removal when size > 1. Zero rows stays empty; one row is untouched.
TEST(SeedHostApiDeleteDuplicate, NoOpWhenZeroOrOneRow) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({7});
	auto tIdx = MakeIntVec({11});
	auto qID = MakeIntVec({2});
	auto qIdx = MakeIntVec({5});
	deleteDuplicateSeeds(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	tID.clear();
	tIdx.clear();
	qID.clear();
	qIdx.clear();
	deleteDuplicateSeeds(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Two adjacent seeds in measure-distance order that fall within (W,G) “near pair” → second removed.
// Rows (0,10,0,5) and (0,11,0,6): same qID/tID; indices differ by 1 on both axes → duplicate pair.
TEST(SeedHostApiDeleteDuplicate, RemovesNearDuplicateOnSortedInput) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({10, 11});
	auto qID = MakeIntVec({0, 0});
	auto qIdx = MakeIntVec({5, 6});
	measureDistanceSorting(tID, tIdx, qID, qIdx);

	deleteDuplicateSeeds(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}

// `deleteSeedHasNotNearPair` drops rows that have no predecessor within (W,G).
// Same tIdx but qID jumps 0→1: second row is not “near” first → isolate removed; first (qID=0) remains.
TEST(SeedHostApiDeleteIsolate, RemovesRowNotNearPreviousInSortedOrder) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({10, 10});
	auto qID = MakeIntVec({0, 1});
	auto qIdx = MakeIntVec({5, 5});
	measureDistanceSorting(tID, tIdx, qID, qIdx);

	deleteSeedHasNotNearPair(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(qID[0], 0);
}

// Effective end position kMer + qIdx must not exceed qLen[qID]. Here 15+86 > 100 → seed dropped.
TEST(SeedHostApiBoundary, OnQueryBoundaryRemovesSeedPastSequenceEnd) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({86}); // kMer(15) + idx(86) = 101 > 100
	onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Query boundary: removal uses strict `>` vs length. kMer+idx == qLen → still valid → row kept.
TEST(SeedHostApiBoundary, OnQueryBoundaryKeepsExactBoundary) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({85}); // kMer(15) + idx(85) = 100 == 100 → keep
	onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}

// Strict inequality: kMer+idx < qLen → interior hit → kept.
TEST(SeedHostApiBoundary, OnQueryBoundaryKeepsInteriorSeed) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({84}); // kMer(15) + idx(84) = 99 < 100
	onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}

// `q_begin` only offsets into `qLengthArray`: effective length is `qLengthArray[qID - q_begin]`.
// qID=10, q_begin=10 → length 1000; kMer+qIdx = 10+991 = 1001 > 1000 → removed.
TEST(SeedHostApiBoundary, OnQueryBoundaryRespectsQBeginOffset) {
	const int kMer = 10;
	const int q_begin = 10;
	auto qLen = MakeIntVec({1000});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({10});
	auto qIdx = MakeIntVec({991}); // kMer(10) + idx(991) = 1001 > 1000
	onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Multi-seed: first seed is interior, second is out-of-bounds → only first survives
TEST(SeedHostApiBoundary, OnQueryBoundaryRetainsOnlyInBoundSeed) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID  = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({0, 0});
	auto qID  = MakeIntVec({0, 0});
	auto qIdx = MakeIntVec({80, 90}); // 15+80=95 ok; 15+90=105 > 100
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(qIdx[0], 80);
}

// Same rule on target: kMer + tIdx must not exceed tLen[tID]. Here 12+190 > 200 → removed.
TEST(SeedHostApiBoundary, OnTargetBoundaryRemovesSeedPastSequenceEnd) {
	const int kMer = 12;
	const int t_begin = 0;
	auto tLen = MakeIntVec({200});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({190}); // kMer(12) + idx(190) = 202 > 200
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({0});
	onTargetBoundary(kMer, t_begin, tLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Target boundary: same strict `>` rule as query — equality kMer+tIdx == tLen keeps the seed.
TEST(SeedHostApiBoundary, OnTargetBoundaryKeepsExactBoundary) {
	const int kMer = 12;
	const int t_begin = 0;
	auto tLen = MakeIntVec({200});
	auto tID  = MakeIntVec({0});
	auto tIdx = MakeIntVec({188}); // kMer(12) + idx(188) = 200 == 200 → keep
	auto qID  = MakeIntVec({0});
	auto qIdx = MakeIntVec({0});
	onTargetBoundary(kMer, t_begin, tLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}

// Multi-seed: one in-bounds, one out-of-bounds; verify correct one is retained
TEST(SeedHostApiBoundary, OnTargetBoundaryRetainsOnlyInBoundSeed) {
	const int kMer = 10;
	const int t_begin = 0;
	auto tLen = MakeIntVec({50});
	auto tID  = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({39, 45}); // 10+39=49 ok; 10+45=55 > 50
	auto qID  = MakeIntVec({0, 0});
	auto qIdx = MakeIntVec({0, 5});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	onTargetBoundary(kMer, t_begin, tLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(tIdx[0], 39);
}

// Lower-left corner: tHitStartIdx + gap < 0
// tHitStartIdx = tIdx - qIdx = 5 - 30 = -25; -25 + 8 = -17 < 0 → remove
TEST(SeedHostApiBoundary, OnCornerRemovesLowerLeftCorner) {
	const int gap = 8;
	const int t_begin = 0;
	const int q_begin = 0;
	auto tLen = MakeIntVec({100});
	auto qLen = MakeIntVec({50});
	auto tID  = MakeIntVec({0});
	auto tIdx = MakeIntVec({5});
	auto qID  = MakeIntVec({0});
	auto qIdx = MakeIntVec({30}); // tHitStartIdx=-25; -25+8=-17 < 0
	onCorner(gap, t_begin, q_begin, tLen, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Upper-right corner: tHitStartIdx + qLength - gap > tLength
// tHitStartIdx = tIdx - qIdx = 80 - 5 = 75; 75 + 50 - 8 = 117 > 100 → remove
TEST(SeedHostApiBoundary, OnCornerRemovesUpperRightCorner) {
	const int gap = 8;
	const int t_begin = 0;
	const int q_begin = 0;
	auto tLen = MakeIntVec({100});
	auto qLen = MakeIntVec({50});
	auto tID  = MakeIntVec({0});
	auto tIdx = MakeIntVec({80});
	auto qID  = MakeIntVec({0});
	auto qIdx = MakeIntVec({5}); // tHitStartIdx=75; 75+50-8=117 > 100
	onCorner(gap, t_begin, q_begin, tLen, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Interior seed: neither corner condition triggered → kept
// tHitStartIdx = 20 - 5 = 15; 15+8=23 >= 0; 15+50-8=57 <= 100
TEST(SeedHostApiBoundary, OnCornerKeepsInteriorSeed) {
	const int gap = 8;
	const int t_begin = 0;
	const int q_begin = 0;
	auto tLen = MakeIntVec({100});
	auto qLen = MakeIntVec({50});
	auto tID  = MakeIntVec({0});
	auto tIdx = MakeIntVec({20});
	auto qID  = MakeIntVec({0});
	auto qIdx = MakeIntVec({5}); // tHitStartIdx=15; 23>=0 and 57<=100
	onCorner(gap, t_begin, q_begin, tLen, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}
