// Unit tests for clast::hit::onQueryBoundary, onTargetBoundary, onCorner
// (src/device/hit/deleteSeedsOnSequenceBoundary.cuh).

#include "device/hit/deleteSeedsOnSequenceBoundary.cuh"
#include "device/hit/sortSeeds.cuh"
#include "hitTestUtil.hpp"

#include <gtest/gtest.h>

using clast::test::hit::MakeIntVec;

// Effective end position kMer + qIdx must not exceed qLen[qID]. Here 15+86 > 100 → seed dropped.
TEST(DeleteSeedsOnSequenceBoundary, OnQueryBoundaryRemovesSeedPastSequenceEnd) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({86}); // kMer(15) + idx(86) = 101 > 100
	clast::hit::onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Query boundary: removal uses strict `>` vs length. kMer+idx == qLen → still valid → row kept.
TEST(DeleteSeedsOnSequenceBoundary, OnQueryBoundaryKeepsExactBoundary) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({85}); // kMer(15) + idx(85) = 100 == 100 → keep
	clast::hit::onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}

// Strict inequality: kMer+idx < qLen → interior hit → kept.
TEST(DeleteSeedsOnSequenceBoundary, OnQueryBoundaryKeepsInteriorSeed) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({84}); // kMer(15) + idx(84) = 99 < 100
	clast::hit::onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}

// `q_begin` only offsets into `qLengthArray`: effective length is `qLengthArray[qID - q_begin]`.
// qID=10, q_begin=10 → length 1000; kMer+qIdx = 10+991 = 1001 > 1000 → removed.
TEST(DeleteSeedsOnSequenceBoundary, OnQueryBoundaryRespectsQBeginOffset) {
	const int kMer = 10;
	const int q_begin = 10;
	auto qLen = MakeIntVec({1000});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({10});
	auto qIdx = MakeIntVec({991}); // kMer(10) + idx(991) = 1001 > 1000
	clast::hit::onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Multi-seed: first seed is interior, second is out-of-bounds → only first survives
TEST(DeleteSeedsOnSequenceBoundary, OnQueryBoundaryRetainsOnlyInBoundSeed) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID  = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({0, 0});
	auto qID  = MakeIntVec({0, 0});
	auto qIdx = MakeIntVec({80, 90}); // 15+80=95 ok; 15+90=105 > 100
	clast::hit::measureDistanceSorting(tID, tIdx, qID, qIdx);
	clast::hit::onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(qIdx[0], 80);
}

// Same rule on target: kMer + tIdx must not exceed tLen[tID]. Here 12+190 > 200 → removed.
TEST(DeleteSeedsOnSequenceBoundary, OnTargetBoundaryRemovesSeedPastSequenceEnd) {
	const int kMer = 12;
	const int t_begin = 0;
	auto tLen = MakeIntVec({200});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({190}); // kMer(12) + idx(190) = 202 > 200
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({0});
	clast::hit::onTargetBoundary(kMer, t_begin, tLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Target boundary: same strict `>` rule as query — equality kMer+tIdx == tLen keeps the seed.
TEST(DeleteSeedsOnSequenceBoundary, OnTargetBoundaryKeepsExactBoundary) {
	const int kMer = 12;
	const int t_begin = 0;
	auto tLen = MakeIntVec({200});
	auto tID  = MakeIntVec({0});
	auto tIdx = MakeIntVec({188}); // kMer(12) + idx(188) = 200 == 200 → keep
	auto qID  = MakeIntVec({0});
	auto qIdx = MakeIntVec({0});
	clast::hit::onTargetBoundary(kMer, t_begin, tLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}

// Multi-seed: one in-bounds, one out-of-bounds; verify correct one is retained
TEST(DeleteSeedsOnSequenceBoundary, OnTargetBoundaryRetainsOnlyInBoundSeed) {
	const int kMer = 10;
	const int t_begin = 0;
	auto tLen = MakeIntVec({50});
	auto tID  = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({39, 45}); // 10+39=49 ok; 10+45=55 > 50
	auto qID  = MakeIntVec({0, 0});
	auto qIdx = MakeIntVec({0, 5});
	clast::hit::measureDistanceSorting(tID, tIdx, qID, qIdx);
	clast::hit::onTargetBoundary(kMer, t_begin, tLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(tIdx[0], 39);
}

// Lower-left corner: tHitStartIdx + gap < 0
// tHitStartIdx = tIdx - qIdx = 5 - 30 = -25; -25 + 8 = -17 < 0 → remove
TEST(DeleteSeedsOnSequenceBoundary, OnCornerRemovesLowerLeftCorner) {
	const int gap = 8;
	const int t_begin = 0;
	const int q_begin = 0;
	auto tLen = MakeIntVec({100});
	auto qLen = MakeIntVec({50});
	auto tID  = MakeIntVec({0});
	auto tIdx = MakeIntVec({5});
	auto qID  = MakeIntVec({0});
	auto qIdx = MakeIntVec({30}); // tHitStartIdx=-25; -25+8=-17 < 0
	clast::hit::onCorner(gap, t_begin, q_begin, tLen, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Upper-right corner: tHitStartIdx + qLength - gap > tLength
// tHitStartIdx = tIdx - qIdx = 80 - 5 = 75; 75 + 50 - 8 = 117 > 100 → remove
TEST(DeleteSeedsOnSequenceBoundary, OnCornerRemovesUpperRightCorner) {
	const int gap = 8;
	const int t_begin = 0;
	const int q_begin = 0;
	auto tLen = MakeIntVec({100});
	auto qLen = MakeIntVec({50});
	auto tID  = MakeIntVec({0});
	auto tIdx = MakeIntVec({80});
	auto qID  = MakeIntVec({0});
	auto qIdx = MakeIntVec({5}); // tHitStartIdx=75; 75+50-8=117 > 100
	clast::hit::onCorner(gap, t_begin, q_begin, tLen, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

// Interior seed: neither corner condition triggered → kept
// tHitStartIdx = 20 - 5 = 15; 15+8=23 >= 0; 15+50-8=57 <= 100
TEST(DeleteSeedsOnSequenceBoundary, OnCornerKeepsInteriorSeed) {
	const int gap = 8;
	const int t_begin = 0;
	const int q_begin = 0;
	auto tLen = MakeIntVec({100});
	auto qLen = MakeIntVec({50});
	auto tID  = MakeIntVec({0});
	auto tIdx = MakeIntVec({20});
	auto qID  = MakeIntVec({0});
	auto qIdx = MakeIntVec({5}); // tHitStartIdx=15; 23>=0 and 57<=100
	clast::hit::onCorner(gap, t_begin, q_begin, tLen, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}
