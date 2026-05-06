// Unit tests for the clast::hit functors and template sort functions declared
// in src/device/hit/alignmentHits.cuh:
//   make_alignmentSizeBackward, alignLengthBackward, alignLengthForward,
//   sortBeforeAlignBackWard, sortBeforeAlignForward.

#include "device/hit/alignmentHits.cuh"
#include "hitTestUtil.hpp"

#include <gtest/gtest.h>

using clast::test::hit::MakeIntVec;

// ─── make_alignmentSizeBackward ──────────────────────────────────────────────

// alignmentSize = queryLength - (queryIndex + qHitLength)
TEST(MakeAlignmentSizeBackward, BasicComputation) {
	clast::hit::make_alignmentSizeBackward f;
	// 100 - (30 + 20) = 50
	EXPECT_EQ(f(thrust::make_tuple(100, 30, 20)), 50);
}

TEST(MakeAlignmentSizeBackward, ZeroWhenHitFillsRemainder) {
	clast::hit::make_alignmentSizeBackward f;
	// 100 - (40 + 60) = 0
	EXPECT_EQ(f(thrust::make_tuple(100, 40, 60)), 0);
}

TEST(MakeAlignmentSizeBackward, FullQueryLength) {
	clast::hit::make_alignmentSizeBackward f;
	// 80 - (0 + 0) = 80
	EXPECT_EQ(f(thrust::make_tuple(80, 0, 0)), 80);
}

// ─── alignLengthBackward ─────────────────────────────────────────────────────

// Comparator reads element 8 (alignmentSize) and returns true when lhs > rhs.
TEST(AlignLengthBackward, LargerSizeComesFirst) {
	clast::hit::alignLengthBackward f;
	// 9-element tuples; only element 8 matters
	auto big   = thrust::make_tuple(0,0,0,0,0,0,0,0,70);
	auto small_ = thrust::make_tuple(0,0,0,0,0,0,0,0,30);
	EXPECT_TRUE (f(big,    small_));
	EXPECT_FALSE(f(small_, big   ));
}

TEST(AlignLengthBackward, EqualSizeReturnsFalse) {
	clast::hit::alignLengthBackward f;
	auto a = thrust::make_tuple(0,0,0,0,0,0,0,0,50);
	auto b = thrust::make_tuple(0,0,0,0,0,0,0,0,50);
	EXPECT_FALSE(f(a, b));
	EXPECT_FALSE(f(b, a));
}

// ─── alignLengthForward ──────────────────────────────────────────────────────

// Comparator reads element 3 (queryIndex) and returns true when lhs > rhs.
TEST(AlignLengthForward, LargerQueryIndexComesFirst) {
	clast::hit::alignLengthForward f;
	// 8-element tuples; only element 3 matters
	auto high = thrust::make_tuple(0,0,0,40,0,0,0,0);
	auto low  = thrust::make_tuple(0,0,0,10,0,0,0,0);
	EXPECT_TRUE (f(high, low ));
	EXPECT_FALSE(f(low,  high));
}

TEST(AlignLengthForward, EqualQueryIndexReturnsFalse) {
	clast::hit::alignLengthForward f;
	auto a = thrust::make_tuple(0,0,0,20,0,0,0,0);
	auto b = thrust::make_tuple(0,0,0,20,0,0,0,0);
	EXPECT_FALSE(f(a, b));
	EXPECT_FALSE(f(b, a));
}

// ─── sortBeforeAlignForward ──────────────────────────────────────────────────

// Empty input: no crash; all arrays stay empty.
TEST(SortBeforeAlignForward, EmptyInputUnchanged) {
	auto tID=MakeIntVec({}), qID=MakeIntVec({}),
	     tIdx=MakeIntVec({}), qIdx=MakeIntVec({}),
	     tLen=MakeIntVec({}), qLen=MakeIntVec({}),
	     mat=MakeIntVec({}),  sc=MakeIntVec({});
	clast::hit::sortBeforeAlignForward(tID,qID,tIdx,qIdx,tLen,qLen,mat,sc);
	ASSERT_EQ(tID.size(), 0u);
}

// Single row: vacuously sorted; all values preserved.
TEST(SortBeforeAlignForward, SingleRowUnchanged) {
	auto tID=MakeIntVec({3}), qID=MakeIntVec({1}),
	     tIdx=MakeIntVec({5}), qIdx=MakeIntVec({20}),
	     tLen=MakeIntVec({50}), qLen=MakeIntVec({40}),
	     mat=MakeIntVec({30}), sc=MakeIntVec({100});
	clast::hit::sortBeforeAlignForward(tID,qID,tIdx,qIdx,tLen,qLen,mat,sc);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(tID[0], 3); EXPECT_EQ(qIdx[0], 20);
}

// Two rows with distinct queryIndex: larger comes first.
// Input:  (tID=0, qIdx=10), (tID=1, qIdx=30)
// Output: (tID=1, qIdx=30), (tID=0, qIdx=10)
TEST(SortBeforeAlignForward, TwoRowsSortedByQueryIndexDescending) {
	auto tID=MakeIntVec({0,1}), qID=MakeIntVec({0,0}),
	     tIdx=MakeIntVec({0,0}), qIdx=MakeIntVec({10,30}),
	     tLen=MakeIntVec({50,50}), qLen=MakeIntVec({40,40}),
	     mat=MakeIntVec({20,20}), sc=MakeIntVec({100,100});
	clast::hit::sortBeforeAlignForward(tID,qID,tIdx,qIdx,tLen,qLen,mat,sc);
	EXPECT_EQ(qIdx[0], 30); EXPECT_EQ(tID[0], 1);
	EXPECT_EQ(qIdx[1], 10); EXPECT_EQ(tID[1], 0);
}

// Three rows: already-reversed input gets correctly sorted.
// queryIdx values 10, 30, 20 → after descending sort: 30, 20, 10.
// tID tracks row identity: 1, 2, 3 → 2, 3, 1.
TEST(SortBeforeAlignForward, ThreeRowsAllArraysMoveTogether) {
	auto tID=MakeIntVec({1,2,3}), qID=MakeIntVec({0,0,0}),
	     tIdx=MakeIntVec({0,0,0}), qIdx=MakeIntVec({10,30,20}),
	     tLen=MakeIntVec({50,50,50}), qLen=MakeIntVec({40,40,40}),
	     mat=MakeIntVec({20,20,20}), sc=MakeIntVec({100,100,100});
	clast::hit::sortBeforeAlignForward(tID,qID,tIdx,qIdx,tLen,qLen,mat,sc);
	EXPECT_EQ(qIdx[0], 30); EXPECT_EQ(tID[0], 2);
	EXPECT_EQ(qIdx[1], 20); EXPECT_EQ(tID[1], 3);
	EXPECT_EQ(qIdx[2], 10); EXPECT_EQ(tID[2], 1);
}

// ─── sortBeforeAlignBackWard ─────────────────────────────────────────────────

// Empty input: no crash; all arrays stay empty.
TEST(SortBeforeAlignBackWard, EmptyInputUnchanged) {
	auto qLen=MakeIntVec({100});
	auto tID=MakeIntVec({}), qID=MakeIntVec({}),
	     tIdx=MakeIntVec({}), qIdx=MakeIntVec({}),
	     tLen=MakeIntVec({}), qLen2=MakeIntVec({}),
	     mat=MakeIntVec({}),  sc=MakeIntVec({});
	clast::hit::sortBeforeAlignBackWard(0,qLen,tID,qID,tIdx,qIdx,tLen,qLen2,mat,sc);
	ASSERT_EQ(tID.size(), 0u);
}

// Single row: vacuously sorted; all values preserved.
// alignmentSize = 100 - (10 + 20) = 70  (does not affect single-row result)
TEST(SortBeforeAlignBackWard, SingleRowUnchanged) {
	auto qLen=MakeIntVec({100});
	auto tID=MakeIntVec({5}), qID=MakeIntVec({0}),
	     tIdx=MakeIntVec({8}), qIdx=MakeIntVec({10}),
	     tLen=MakeIntVec({50}), qLen2=MakeIntVec({20}),
	     mat=MakeIntVec({15}), sc=MakeIntVec({80});
	clast::hit::sortBeforeAlignBackWard(0,qLen,tID,qID,tIdx,qIdx,tLen,qLen2,mat,sc);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(tID[0], 5); EXPECT_EQ(qIdx[0], 10);
}

// Two rows: larger alignmentSize comes first.
// qLengthArray = {100}, q_begin = 0.
// Row A (tID=0): queryID=0, queryIdx=10, qHitLen=20  → alignSize = 100-(10+20) = 70
// Row B (tID=1): queryID=0, queryIdx=40, qHitLen=30  → alignSize = 100-(40+30) = 30
// Input order: B then A.  After sort: A (70) first, B (30) second.
TEST(SortBeforeAlignBackWard, TwoRowsSortedByAlignmentSizeDescending) {
	auto qLen=MakeIntVec({100});
	//                tID   qID  tIdx   qIdx  tLen  qHitLen  mat   sc
	auto tID=MakeIntVec({1,  0}),  qID=MakeIntVec({0,  0}),
	     tIdx=MakeIntVec({20, 5}), qIdx=MakeIntVec({40, 10}),
	     tLen=MakeIntVec({50,50}), qLen2=MakeIntVec({30, 20}),
	     mat=MakeIntVec({25,20}),  sc=MakeIntVec({150,100});
	clast::hit::sortBeforeAlignBackWard(0,qLen,tID,qID,tIdx,qIdx,tLen,qLen2,mat,sc);
	// Row A (tID=0) has larger alignSize (70 > 30) → first after sort
	EXPECT_EQ(tID[0], 0); EXPECT_EQ(qIdx[0], 10);
	EXPECT_EQ(tID[1], 1); EXPECT_EQ(qIdx[1], 40);
}

// Three rows with distinct alignmentSizes; all eight arrays move together.
// qLengthArray = {100}, q_begin = 0.
// Row A (tID=0): queryIdx=10, qHitLen=10  → alignSize = 100-20 = 80  (largest)
// Row B (tID=1): queryIdx=30, qHitLen=20  → alignSize = 100-50 = 50
// Row C (tID=2): queryIdx=60, qHitLen=30  → alignSize = 100-90 = 10  (smallest)
// Input order reversed: C, B, A.  Expected after sort: A, B, C.
TEST(SortBeforeAlignBackWard, ThreeRowsAllArraysMoveTogether) {
	auto qLen=MakeIntVec({100});
	auto tID=MakeIntVec({2,  1,  0}), qID=MakeIntVec({0,0,0}),
	     tIdx=MakeIntVec({0,  0,  0}), qIdx=MakeIntVec({60, 30, 10}),
	     tLen=MakeIntVec({50,50,50}),  qLen2=MakeIntVec({30, 20, 10}),
	     mat=MakeIntVec({20,20,20}),   sc=MakeIntVec({100,100,100});
	clast::hit::sortBeforeAlignBackWard(0,qLen,tID,qID,tIdx,qIdx,tLen,qLen2,mat,sc);
	EXPECT_EQ(tID[0], 0); EXPECT_EQ(qIdx[0], 10); // alignSize 80
	EXPECT_EQ(tID[1], 1); EXPECT_EQ(qIdx[1], 30); // alignSize 50
	EXPECT_EQ(tID[2], 2); EXPECT_EQ(qIdx[2], 60); // alignSize 10
}

// q_begin > 0: queryID values are offset; each queryID[i] - q_begin indexes qLengthArray.
// qLengthArray = {100}, q_begin = 5.
// Row A (tID=0): queryID=5, queryIdx=10, qHitLen=20 → alignSize = qLen[5-5]-(10+20) = 70
// Row B (tID=1): queryID=5, queryIdx=40, qHitLen=30 → alignSize = 100-(40+30) = 30
// Input: B then A.  After sort: A first.
TEST(SortBeforeAlignBackWard, QBeginOffsetCorrectlyApplied) {
	auto qLen=MakeIntVec({100}); // qLen[0] = 100; accessed as qLen[queryID - q_begin]
	auto tID=MakeIntVec({1,  0}), qID=MakeIntVec({5,  5}),
	     tIdx=MakeIntVec({20, 5}), qIdx=MakeIntVec({40, 10}),
	     tLen=MakeIntVec({50,50}), qLen2=MakeIntVec({30, 20}),
	     mat=MakeIntVec({25,20}),  sc=MakeIntVec({150,100});
	clast::hit::sortBeforeAlignBackWard(5,qLen,tID,qID,tIdx,qIdx,tLen,qLen2,mat,sc);
	EXPECT_EQ(tID[0], 0); // alignSize 70
	EXPECT_EQ(tID[1], 1); // alignSize 30
}
