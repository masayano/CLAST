// Unit tests for `src/util/utilResultSorting.cuh`:
// `clast::sorting::result` comparator and `clast::sorting::resultSorting` template.

#include "util/utilResultSorting.cuh"

#include <gtest/gtest.h>

#include <thrust/host_vector.h>
#include <thrust/tuple.h>

// ─── helpers ────────────────────────────────────────────────────────────────

// Build a tuple with the fields that matter for the comparator;
// unused fields (targetID, targetIndex, queryIndex, tHitLength, qHitLength,
// matchNum, evalue) are filled with 0.
// tuple<int,int,int,int,int,int,int,int,double>
//       [0] [1] [2] [3] [4] [5] [6] [7] [8]
//  targetID tIdx qID  qIdx tLen qLen match scr  eval
static thrust::tuple<int,int,int,int,int,int,int,int,double>
makeHit(int queryID, int score) {
	return thrust::make_tuple(0, 0, queryID, 0, 0, 0, 0, score, 0.0);
}

// ─── result comparator ──────────────────────────────────────────────────────

// Lower queryID/2 comes first.
TEST(ResultComparator, LowerQIDFirst) {
	clast::sorting::result cmp;
	// queryID 0 (strand 0) → qID/2 = 0; queryID 2 (strand 0) → qID/2 = 1
	EXPECT_TRUE (cmp(makeHit(0, 10), makeHit(2, 10)));
	EXPECT_FALSE(cmp(makeHit(2, 10), makeHit(0, 10)));
}

// Within the same qID/2 group, higher score comes first.
TEST(ResultComparator, HigherScoreFirstWithinSameQuery) {
	clast::sorting::result cmp;
	EXPECT_TRUE (cmp(makeHit(0, 20), makeHit(0, 10)));
	EXPECT_FALSE(cmp(makeHit(0, 10), makeHit(0, 20)));
}

// Equal score within same group → not-less-than in both directions (strict weak ordering).
TEST(ResultComparator, EqualScoreSameQuery_NotLessThan) {
	clast::sorting::result cmp;
	EXPECT_FALSE(cmp(makeHit(0, 10), makeHit(0, 10)));
}

// Strand pair collapse: queryID 0 and 1 both map to group 0 (qID/2 == 0).
TEST(ResultComparator, StrandPairCollapse_OddAndEvenSameGroup) {
	clast::sorting::result cmp;
	// queryID 0 and 1 are in the same group; neither should come "before" the other
	// based on group ID (they're equal), but score may differ.
	auto a = makeHit(0, 5);
	auto b = makeHit(1, 5);
	// Same group (0/2==0, 1/2==0), same score → both false
	EXPECT_FALSE(cmp(a, b));
	EXPECT_FALSE(cmp(b, a));
}

// Different groups, strand-pair: group 0 (qID=1) < group 1 (qID=2).
TEST(ResultComparator, StrandPairCollapse_Group0BeforeGroup1) {
	clast::sorting::result cmp;
	EXPECT_TRUE (cmp(makeHit(1, 10), makeHit(2, 10)));
	EXPECT_FALSE(cmp(makeHit(2, 10), makeHit(1, 10)));
}

// ─── clast::sorting::resultSorting template ─────────────────────────────────

using IntVec = thrust::host_vector<int>;
using DblVec = thrust::host_vector<double>;

// Empty arrays: no crash, arrays remain empty.
TEST(ResultSorting, EmptyArrays) {
	IntVec tID, tIdx, qID, qIdx, tLen, qLen, match, scr;
	DblVec eval;
	EXPECT_NO_THROW(clast::sorting::resultSorting(
			tID, tIdx, qID, qIdx, tLen, qLen, match, scr, eval));
	EXPECT_EQ(tID.size(), 0u);
}

// Single row: unchanged after sort.
TEST(ResultSorting, SingleRow_Unchanged) {
	IntVec tID{7}, tIdx{1}, qID{4}, qIdx{2}, tLen{3}, qLen{5}, match{6}, scr{99};
	DblVec eval{1.5};
	clast::sorting::resultSorting(tID, tIdx, qID, qIdx, tLen, qLen, match, scr, eval);
	EXPECT_EQ(qID[0], 4);
	EXPECT_EQ(scr[0], 99);
	EXPECT_EQ(tID[0], 7);
}

// Two rows, already sorted (lower qID first, higher score first): order preserved.
TEST(ResultSorting, TwoRows_AlreadySorted) {
	//           row0: qID=0, scr=20   row1: qID=2, scr=10
	IntVec tID{0,0}, tIdx{0,0}, qID{0,2}, qIdx{0,0}, tLen{0,0},
	       qLen{0,0}, match{0,0}, scr{20,10};
	DblVec eval{0.0, 0.0};
	clast::sorting::resultSorting(tID, tIdx, qID, qIdx, tLen, qLen, match, scr, eval);
	EXPECT_EQ(qID[0], 0);  EXPECT_EQ(scr[0], 20);
	EXPECT_EQ(qID[1], 2);  EXPECT_EQ(scr[1], 10);
}

// Two rows, reversed qID: reordered so lower qID comes first.
TEST(ResultSorting, TwoRows_SortsByQID) {
	IntVec tID{0,0}, tIdx{0,0}, qID{4,0}, qIdx{0,0}, tLen{0,0},
	       qLen{0,0}, match{0,0}, scr{10,20};
	DblVec eval{0.0, 0.0};
	clast::sorting::resultSorting(tID, tIdx, qID, qIdx, tLen, qLen, match, scr, eval);
	EXPECT_EQ(qID[0], 0);
	EXPECT_EQ(qID[1], 4);
}

// Within the same qID group: higher score first.
TEST(ResultSorting, SameQID_SortsByScoreDescending) {
	IntVec tID{0,0}, tIdx{0,0}, qID{0,0}, qIdx{0,0}, tLen{0,0},
	       qLen{0,0}, match{0,0}, scr{5,30};
	DblVec eval{0.0, 0.0};
	clast::sorting::resultSorting(tID, tIdx, qID, qIdx, tLen, qLen, match, scr, eval);
	EXPECT_EQ(scr[0], 30);
	EXPECT_EQ(scr[1], 5);
}

// All arrays are rearranged together (check tID tracks with qID).
TEST(ResultSorting, AllArraysMoveTogether) {
	IntVec tID{99,77}, tIdx{0,0}, qID{4,0}, qIdx{0,0}, tLen{0,0},
	       qLen{0,0}, match{0,0}, scr{10,20};
	DblVec eval{1.1, 2.2};
	clast::sorting::resultSorting(tID, tIdx, qID, qIdx, tLen, qLen, match, scr, eval);
	// row for qID=0 moves to front; its tID was 77 and eval was 2.2
	EXPECT_EQ(qID[0], 0);
	EXPECT_EQ(tID[0], 77);
	EXPECT_DOUBLE_EQ(eval[0], 2.2);
}

// Four rows across two qID groups; verify full ordering.
TEST(ResultSorting, FourRows_TwoGroups) {
	// Group 0 (qID 0,1): scores 5, 15 → after: 15,5
	// Group 1 (qID 2,3): scores 8, 12 → after: 12,8
	IntVec tID{0,0,0,0}, tIdx{0,0,0,0},
	       qID{0,1,2,3},
	       qIdx{0,0,0,0}, tLen{0,0,0,0}, qLen{0,0,0,0},
	       match{0,0,0,0},
	       scr{5,15,8,12};
	DblVec eval{0,0,0,0};
	clast::sorting::resultSorting(tID, tIdx, qID, qIdx, tLen, qLen, match, scr, eval);
	// Group 0 rows first (qID/2 == 0), group 1 rows second (qID/2 == 1).
	// Within group 0: higher score first → scr 15 before 5.
	// Within group 1: higher score first → scr 12 before 8.
	EXPECT_EQ(scr[0], 15); EXPECT_EQ(scr[1], 5);
	EXPECT_EQ(scr[2], 12); EXPECT_EQ(scr[3], 8);
}
