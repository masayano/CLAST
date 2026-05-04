// Unit tests for host-side Thrust seed helpers in `src/device/hit/seedHostApi.cu`
// (duplicate / isolate / boundary / corner), plus `sortSeeds` where needed to build input.

#include "device/hit/seedHostApi.cuh"
#include "device/hit/sortSeeds.cuh"
#include "hitSortSeedsExpect.hpp"

#include <gtest/gtest.h>

using clast::test::hit::ExpectSortedByMeasureDistance;
using clast::test::hit::MakeIntVec;

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

TEST(SeedHostApiDeleteDuplicate, RemovesNearDuplicateOnSortedInput) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({10, 11});
	auto qID = MakeIntVec({0, 0});
	auto qIdx = MakeIntVec({5, 6});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	deleteDuplicateSeeds(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}

TEST(SeedHostApiDeleteIsolate, RemovesRowNotNearPreviousInSortedOrder) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({10, 10});
	auto qID = MakeIntVec({0, 1});
	auto qIdx = MakeIntVec({5, 5});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	deleteSeedHasNotNearPair(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(qID[0], 0);
}

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

// Exact boundary: kMer + idx == length → kept (is_onBoundary uses strict >)
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

// Exact boundary: kMer + tIdx == tLength → kept
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
