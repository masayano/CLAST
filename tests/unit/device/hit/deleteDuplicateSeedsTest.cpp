// Unit tests for clast::hit::deleteDuplicateSeeds (src/device/hit/deleteDuplicateSeeds.cuh).

#include "device/hit/deleteDuplicateSeeds.cuh"
#include "device/hit/sortSeeds.cuh"
#include "hitTestUtil.hpp"

#include <gtest/gtest.h>

using clast::test::hit::MakeIntVec;

// `deleteDuplicateSeeds` only runs removal when size > 1. Zero rows stays empty; one row is untouched.
TEST(DeleteDuplicateSeeds, NoOpWhenZeroOrOneRow) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({7});
	auto tIdx = MakeIntVec({11});
	auto qID = MakeIntVec({2});
	auto qIdx = MakeIntVec({5});
	clast::hit::deleteDuplicateSeeds(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	tID.clear();
	tIdx.clear();
	qID.clear();
	qIdx.clear();
	clast::hit::deleteDuplicateSeeds(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Two adjacent seeds in measure-distance order that fall within (W,G) "near pair" → second removed.
// Rows (0,10,0,5) and (0,11,0,6): same qID/tID; indices differ by 1 on both axes → duplicate pair.
TEST(DeleteDuplicateSeeds, RemovesNearDuplicateOnSortedInput) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({10, 11});
	auto qID = MakeIntVec({0, 0});
	auto qIdx = MakeIntVec({5, 6});
	clast::hit::measureDistanceSorting(tID, tIdx, qID, qIdx);

	clast::hit::deleteDuplicateSeeds(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}
