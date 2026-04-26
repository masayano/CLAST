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
	auto qIdx = MakeIntVec({86}); // kMer + idx > length
	onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

TEST(SeedHostApiBoundary, OnQueryBoundaryKeepsInteriorSeed) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({84}); // kMer + idx == 99 <= 100
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
	auto qIdx = MakeIntVec({991}); // maps to length entry at index 0
	onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

TEST(SeedHostApiBoundary, OnTargetBoundaryRemovesSeedPastSequenceEnd) {
	const int kMer = 12;
	const int t_begin = 0;
	auto tLen = MakeIntVec({200});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({190});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({0});
	onTargetBoundary(kMer, t_begin, tLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

TEST(SeedHostApiBoundary, OnCornerRemovesWhenHitPatchCrossesCorners) {
	const int gap = 8;
	const int t_begin = 0;
	const int q_begin = 0;
	auto tLen = MakeIntVec({100});
	auto qLen = MakeIntVec({50});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({5});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({30}); // tHitStartIdx = -25 ; +gap < 0
	onCorner(gap, t_begin, q_begin, tLen, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}
