// Unit tests for clast::hit::removeLowEValue and clast::hit::removeTooShort
// (src/device/hit/deleteBadHits.cuh).

#include "device/hit/deleteBadHits.cuh"
#include "hitTestUtil.hpp"

#include <gtest/gtest.h>

using clast::test::hit::MakeIntVec;
using clast::test::hit::MakeDoubleVec;

// Helper: build nine hit arrays with a single row.
// targetID=0, targetIdx=0, queryID=0, queryIdx=0, tLen=50, qLen=50, match=40, score=100
// Caller supplies only the field under test; the rest are fixed dummies.
static void MakeSingleHit(
		thrust::host_vector<int>&    targetID,
		thrust::host_vector<int>&    targetIdx,
		thrust::host_vector<int>&    queryID,
		thrust::host_vector<int>&    queryIdx,
		thrust::host_vector<int>&    tLen,
		thrust::host_vector<int>&    qLen,
		thrust::host_vector<int>&    match,
		thrust::host_vector<int>&    score,
		thrust::host_vector<double>& evalue,
		int qLenVal, double evalueVal) {
	targetID  = MakeIntVec({0});
	targetIdx = MakeIntVec({0});
	queryID   = MakeIntVec({0});
	queryIdx  = MakeIntVec({0});
	tLen      = MakeIntVec({50});
	qLen      = MakeIntVec({qLenVal});
	match     = MakeIntVec({40});
	score     = MakeIntVec({100});
	evalue    = MakeDoubleVec({evalueVal});
}

// ─── removeLowEValue ─────────────────────────────────────────────────────────

// Removal condition: cutOffEValue < eValue (strict). eValue > cutoff → removed.
TEST(DeleteBadHits, RemoveLowEValue_RemovesHitWithEValueAboveCutoff) {
	const double cutoff = 1e-5;
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeSingleHit(tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue,
	              /*qLen=*/50, /*evalue=*/1e-3); // 1e-3 > 1e-5 → remove
	clast::hit::removeLowEValue(cutoff, tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	ASSERT_EQ(tID.size(), 0u);
}

// Boundary: cutOffEValue == eValue → condition (cutOff < eValue) is false → kept.
TEST(DeleteBadHits, RemoveLowEValue_KeepsHitWithEValueEqualToCutoff) {
	const double cutoff = 1e-5;
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeSingleHit(tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue,
	              /*qLen=*/50, /*evalue=*/1e-5); // 1e-5 == 1e-5 → keep
	clast::hit::removeLowEValue(cutoff, tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	ASSERT_EQ(tID.size(), 1u);
}

// Interior: eValue < cutoff → kept.
TEST(DeleteBadHits, RemoveLowEValue_KeepsHitWithEValueBelowCutoff) {
	const double cutoff = 1e-5;
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeSingleHit(tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue,
	              /*qLen=*/50, /*evalue=*/1e-10); // 1e-10 < 1e-5 → keep
	clast::hit::removeLowEValue(cutoff, tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	ASSERT_EQ(tID.size(), 1u);
}

// Multi-hit: one above cutoff (queryID=1), one below (queryID=0) → only below survives.
TEST(DeleteBadHits, RemoveLowEValue_RetainsOnlyBelowCutoffHit) {
	const double cutoff = 1e-5;
	auto tID   = MakeIntVec({0, 0});
	auto tIdx  = MakeIntVec({0, 0});
	auto qID   = MakeIntVec({0, 1}); // discriminator
	auto qIdx  = MakeIntVec({0, 0});
	auto tLen  = MakeIntVec({50, 50});
	auto qLen  = MakeIntVec({50, 50});
	auto match = MakeIntVec({40, 40});
	auto score = MakeIntVec({100, 100});
	auto ev    = MakeDoubleVec({1e-10, 1e-3}); // qID=0 → keep; qID=1 → remove
	clast::hit::removeLowEValue(cutoff, tID, tIdx, qID, qIdx, tLen, qLen, match, score, ev);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(qID[0], 0);
	EXPECT_DOUBLE_EQ(ev[0], 1e-10);
}

// ─── removeTooShort ──────────────────────────────────────────────────────────

// Removal condition: cutOffLength > qHitLength (strict). qLen < cutoff → removed.
TEST(DeleteBadHits, RemoveTooShort_RemovesHitWithQHitLengthBelowCutoff) {
	const int cutoff = 30;
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeSingleHit(tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue,
	              /*qLen=*/20, /*evalue=*/1e-10); // 20 < 30 → remove
	clast::hit::removeTooShort(cutoff, tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	ASSERT_EQ(tID.size(), 0u);
}

// Boundary: qHitLength == cutoff → condition (cutOff > hitLength) is false → kept.
TEST(DeleteBadHits, RemoveTooShort_KeepsHitWithQHitLengthEqualToCutoff) {
	const int cutoff = 30;
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeSingleHit(tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue,
	              /*qLen=*/30, /*evalue=*/1e-10); // 30 == 30 → keep
	clast::hit::removeTooShort(cutoff, tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	ASSERT_EQ(tID.size(), 1u);
}

// Interior: qHitLength > cutoff → kept.
TEST(DeleteBadHits, RemoveTooShort_KeepsHitWithQHitLengthAboveCutoff) {
	const int cutoff = 30;
	thrust::host_vector<int>    tID, tIdx, qID, qIdx, tLen, qLen, match, score;
	thrust::host_vector<double> evalue;
	MakeSingleHit(tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue,
	              /*qLen=*/50, /*evalue=*/1e-10); // 50 > 30 → keep
	clast::hit::removeTooShort(cutoff, tID, tIdx, qID, qIdx, tLen, qLen, match, score, evalue);
	ASSERT_EQ(tID.size(), 1u);
}

// Multi-hit: one too short (queryID=1, qLen=20), one ok (queryID=0, qLen=50) → only ok survives.
TEST(DeleteBadHits, RemoveTooShort_RetainsOnlyLongEnoughHit) {
	const int cutoff = 30;
	auto tID   = MakeIntVec({0, 0});
	auto tIdx  = MakeIntVec({0, 0});
	auto qID   = MakeIntVec({0, 1}); // discriminator
	auto qIdx  = MakeIntVec({0, 0});
	auto tLen  = MakeIntVec({50, 50});
	auto qLen  = MakeIntVec({50, 20}); // qID=0 → keep; qID=1 → remove
	auto match = MakeIntVec({40, 15});
	auto score = MakeIntVec({100, 50});
	auto ev    = MakeDoubleVec({1e-10, 1e-10});
	clast::hit::removeTooShort(cutoff, tID, tIdx, qID, qIdx, tLen, qLen, match, score, ev);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(qID[0], 0);
	EXPECT_EQ(qLen[0], 50);
}
