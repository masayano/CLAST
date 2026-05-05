// Unit tests for clast::hit::is_duplicate and clast::hit::removeDuplicateResultHits
// (src/device/hit/CDeviceHitList.cuh).
//
// removeDuplicateResultHits removes consecutive result-hit rows whose
// (targetID, targetIdx, queryID, queryIdx) all match the preceding row.
// The first occurrence is always preserved.

#include "device/hit/CDeviceHitList.cuh"
#include "hitTestUtil.hpp"

#include <gtest/gtest.h>

using clast::test::hit::MakeIntVec;
using clast::test::hit::MakeDoubleVec;

// ─── is_duplicate ────────────────────────────────────────────────────────────

// All four key fields identical → duplicate.
TEST(IsDuplicate, ReturnsTrueWhenAllFieldsMatch) {
	clast::hit::is_duplicate f;
	auto t = thrust::make_tuple(0, 5, 1, 3,   0, 5, 1, 3);
	EXPECT_TRUE(f(t));
}

// targetID differs → not a duplicate.
TEST(IsDuplicate, ReturnsFalseWhenTargetIDDiffers) {
	clast::hit::is_duplicate f;
	auto t = thrust::make_tuple(1, 5, 1, 3,   0, 5, 1, 3);
	EXPECT_FALSE(f(t));
}

// targetIdx differs → not a duplicate.
TEST(IsDuplicate, ReturnsFalseWhenTargetIdxDiffers) {
	clast::hit::is_duplicate f;
	auto t = thrust::make_tuple(0, 6, 1, 3,   0, 5, 1, 3);
	EXPECT_FALSE(f(t));
}

// queryID differs → not a duplicate.
TEST(IsDuplicate, ReturnsFalseWhenQueryIDDiffers) {
	clast::hit::is_duplicate f;
	auto t = thrust::make_tuple(0, 5, 2, 3,   0, 5, 1, 3);
	EXPECT_FALSE(f(t));
}

// queryIdx differs → not a duplicate.
TEST(IsDuplicate, ReturnsFalseWhenQueryIdxDiffers) {
	clast::hit::is_duplicate f;
	auto t = thrust::make_tuple(0, 5, 1, 4,   0, 5, 1, 3);
	EXPECT_FALSE(f(t));
}

// ─── removeDuplicateResultHits ───────────────────────────────────────────────

// Helper: build all nine hit arrays from parallel initializer lists.
static void MakeHitArrays(
		std::initializer_list<int>    tIDs,
		std::initializer_list<int>    tIdxs,
		std::initializer_list<int>    qIDs,
		std::initializer_list<int>    qIdxs,
		std::initializer_list<int>    tLens,
		std::initializer_list<int>    qLens,
		std::initializer_list<int>    matches,
		std::initializer_list<int>    scores,
		std::initializer_list<double> evalues,
		thrust::host_vector<int>&    tID,
		thrust::host_vector<int>&    tIdx,
		thrust::host_vector<int>&    qID,
		thrust::host_vector<int>&    qIdx,
		thrust::host_vector<int>&    tLen,
		thrust::host_vector<int>&    qLen,
		thrust::host_vector<int>&    match,
		thrust::host_vector<int>&    score,
		thrust::host_vector<double>& evalue) {
	tID   = MakeIntVec(tIDs);
	tIdx  = MakeIntVec(tIdxs);
	qID   = MakeIntVec(qIDs);
	qIdx  = MakeIntVec(qIdxs);
	tLen  = MakeIntVec(tLens);
	qLen  = MakeIntVec(qLens);
	match = MakeIntVec(matches);
	score = MakeIntVec(scores);
	evalue = MakeDoubleVec(evalues);
}

// 0 rows: size-guard (> 1) → no-op; stays empty.
TEST(RemoveDuplicateResultHits, EmptyInputUnchanged) {
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeHitArrays({},{},{},{},{},{},{},{},{},
	              tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	clast::hit::removeDuplicateResultHits(tID,tIdx,qID,qIdx,tLen,qLen,match,score,evalue);
	ASSERT_EQ(tID.size(), 0u);
}

// 1 row: size-guard (> 1) → no-op; single row is kept.
TEST(RemoveDuplicateResultHits, SingleRowUnchanged) {
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeHitArrays({0},{5},{1},{3},{50},{40},{35},{100},{1e-10},
	              tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	clast::hit::removeDuplicateResultHits(tID,tIdx,qID,qIdx,tLen,qLen,match,score,evalue);
	ASSERT_EQ(tID.size(), 1u);
}

// 2 identical rows → first kept, second removed.
TEST(RemoveDuplicateResultHits, TwoIdenticalRowsYieldsOne) {
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeHitArrays({0,0},{5,5},{1,1},{3,3},{50,50},{40,40},{35,35},{100,100},{1e-10,1e-10},
	              tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	clast::hit::removeDuplicateResultHits(tID,tIdx,qID,qIdx,tLen,qLen,match,score,evalue);
	ASSERT_EQ(tID.size(), 1u);
}

// 2 distinct rows → both kept.
TEST(RemoveDuplicateResultHits, TwoDistinctRowsBothKept) {
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeHitArrays({0,0},{5,6},{1,1},{3,3},{50,50},{40,40},{35,35},{100,100},{1e-10,1e-10},
	              tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	clast::hit::removeDuplicateResultHits(tID,tIdx,qID,qIdx,tLen,qLen,match,score,evalue);
	ASSERT_EQ(tID.size(), 2u);
}

// [A, A, B] → [A, B]: first duplicate removed, distinct row kept.
TEST(RemoveDuplicateResultHits, LeadingDuplicateThenDistinct) {
	// Rows: (0,5,1,3), (0,5,1,3), (0,6,1,3) — second is dup of first; third is distinct.
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeHitArrays({0,0,0},{5,5,6},{1,1,1},{3,3,3},{50,50,50},{40,40,40},{35,35,35},{100,100,100},{1e-10,1e-10,1e-10},
	              tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	clast::hit::removeDuplicateResultHits(tID,tIdx,qID,qIdx,tLen,qLen,match,score,evalue);
	ASSERT_EQ(tID.size(), 2u);
	EXPECT_EQ(tIdx[0], 5);
	EXPECT_EQ(tIdx[1], 6);
}

// [A, B, B] → [A, B]: distinct first row, trailing duplicate removed.
TEST(RemoveDuplicateResultHits, DistinctThenTrailingDuplicate) {
	// Rows: (0,5,1,3), (0,6,1,3), (0,6,1,3)
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeHitArrays({0,0,0},{5,6,6},{1,1,1},{3,3,3},{50,50,50},{40,40,40},{35,35,35},{100,100,100},{1e-10,1e-10,1e-10},
	              tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	clast::hit::removeDuplicateResultHits(tID,tIdx,qID,qIdx,tLen,qLen,match,score,evalue);
	ASSERT_EQ(tID.size(), 2u);
	EXPECT_EQ(tIdx[0], 5);
	EXPECT_EQ(tIdx[1], 6);
}

// [A, A, A] → [A]: all three identical, only first survives.
TEST(RemoveDuplicateResultHits, AllIdenticalYieldsOne) {
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeHitArrays({0,0,0},{5,5,5},{1,1,1},{3,3,3},{50,50,50},{40,40,40},{35,35,35},{100,100,100},{1e-10,1e-10,1e-10},
	              tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	clast::hit::removeDuplicateResultHits(tID,tIdx,qID,qIdx,tLen,qLen,match,score,evalue);
	ASSERT_EQ(tID.size(), 1u);
}

// Non-key fields of the surviving row match the FIRST occurrence.
// Row 0 and Row 1 share the same key but have distinct non-key values.
// After dedup, the survivor must carry Row 0's non-key data.
TEST(RemoveDuplicateResultHits, SurvivorCarriesFirstOccurrenceData) {
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	//                 tID  tIdx  qID  qIdx  tLen      qLen      match     score     evalue
	MakeHitArrays({0,  0 },{5, 5},{1, 1},{3, 3},
	              {50, 99},{40,99},{35,99},{100,999},{1e-10, 1e-99},
	              tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	clast::hit::removeDuplicateResultHits(tID,tIdx,qID,qIdx,tLen,qLen,match,score,evalue);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(tLen[0],  50);
	EXPECT_EQ(qLen[0],  40);
	EXPECT_EQ(match[0], 35);
	EXPECT_EQ(score[0], 100);
	EXPECT_DOUBLE_EQ(evalue[0], 1e-10);
}

// Mixed sequence [A, A, B, C, C] → [A, B, C].
TEST(RemoveDuplicateResultHits, MixedSequenceDeduped) {
	// tIdx used as discriminator: 5,5,6,7,7
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeHitArrays({0,0,0,0,0},{5,5,6,7,7},{1,1,1,1,1},{3,3,3,3,3},
	              {50,50,50,50,50},{40,40,40,40,40},{35,35,35,35,35},
	              {100,100,100,100,100},{1e-10,1e-10,1e-10,1e-10,1e-10},
	              tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	clast::hit::removeDuplicateResultHits(tID,tIdx,qID,qIdx,tLen,qLen,match,score,evalue);
	ASSERT_EQ(tID.size(), 3u);
	EXPECT_EQ(tIdx[0], 5);
	EXPECT_EQ(tIdx[1], 6);
	EXPECT_EQ(tIdx[2], 7);
}
