// Unit tests for clast::hit::deleteSeedHasNotNearPair (src/device/hit/deleteIsolateSeeds.cuh).

#include "device/hit/deleteIsolateSeeds.cuh"
#include "device/hit/sortSeeds.cuh"
#include "hitTestUtil.hpp"

#include <gtest/gtest.h>

using clast::test::hit::MakeIntVec;

// `deleteSeedHasNotNearPair` drops rows that have no predecessor within (W,G).
// Same tIdx but qID jumps 0→1: second row is not "near" first → isolate removed; first (qID=0) remains.
TEST(DeleteIsolateSeeds, RemovesRowNotNearPreviousInSortedOrder) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({10, 10});
	auto qID = MakeIntVec({0, 1});
	auto qIdx = MakeIntVec({5, 5});
	clast::hit::measureDistanceSorting(tID, tIdx, qID, qIdx);

	clast::hit::deleteSeedHasNotNearPair(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(qID[0], 0);
}
