// Unit tests for `src/util/utilAddSequence.cu` / `utilAddSequence.cuh`:
// `addSequence` — accumulates index, ID, and base arrays across FASTA sequences.

#include "util/utilAddSequence.cuh"

#include <gtest/gtest.h>

#include <thrust/host_vector.h>
#include <string>
#include <vector>

// ─── helpers ────────────────────────────────────────────────────────────────

static std::vector<int>  toVec(const thrust::host_vector<int>&  v) { return {v.begin(), v.end()}; }
static std::vector<char> toVec(const thrust::host_vector<char>& v) { return {v.begin(), v.end()}; }

// ─── first call (empty arrays) ───────────────────────────────────────────────

// indexArray receives 0-based sequential positions [0 .. seqLength + jointLength - 1].
TEST(AddSequenceFirst, IndexArray_SequentialFromZero) {
	thrust::host_vector<int>  idx, id;
	thrust::host_vector<char> bases;
	addSequence(4, 3, "ACGT", idx, id, bases);
	// jointLength = 3 - 1 = 2; size = 4 + 2 = 6
	EXPECT_EQ(toVec(idx), (std::vector<int>{0, 1, 2, 3, 4, 5}));
}

// IDArray is all zeros for the first sequence.
TEST(AddSequenceFirst, IDArray_AllZeros) {
	thrust::host_vector<int>  idx, id;
	thrust::host_vector<char> bases;
	addSequence(4, 3, "ACGT", idx, id, bases);
	EXPECT_EQ(toVec(id), (std::vector<int>{0, 0, 0, 0, 0, 0}));
}

// baseArray holds the full sequence followed by the first jointLength characters (circular overlap).
TEST(AddSequenceFirst, BaseArray_SeqThenOverlap) {
	thrust::host_vector<int>  idx, id;
	thrust::host_vector<char> bases;
	addSequence(4, 3, "ACGT", idx, id, bases);
	// "ACGT" + "AC" (first jointLength=2 chars)
	EXPECT_EQ(toVec(bases), (std::vector<char>{'A', 'C', 'G', 'T', 'A', 'C'}));
}

// lMerLength = 1 → jointLength = 0: no overlap appended; sizes equal seqLength.
TEST(AddSequenceFirst, NoOverlapWhenLMerLengthIsOne) {
	thrust::host_vector<int>  idx, id;
	thrust::host_vector<char> bases;
	addSequence(3, 1, "ACG", idx, id, bases);
	EXPECT_EQ(toVec(idx),   (std::vector<int> {0, 1, 2}));
	EXPECT_EQ(toVec(id),    (std::vector<int> {0, 0, 0}));
	EXPECT_EQ(toVec(bases), (std::vector<char>{'A', 'C', 'G'}));
}

// Single-character sequence: size = 1 + jointLength; overlap repeats the single char.
TEST(AddSequenceFirst, SingleCharSequence) {
	thrust::host_vector<int>  idx, id;
	thrust::host_vector<char> bases;
	addSequence(1, 2, "A", idx, id, bases);
	// jointLength = 1; size = 2
	EXPECT_EQ(toVec(idx),   (std::vector<int> {0, 1}));
	EXPECT_EQ(toVec(id),    (std::vector<int> {0, 0}));
	EXPECT_EQ(toVec(bases), (std::vector<char>{'A', 'A'}));
}

// ─── second call (non-empty arrays) ─────────────────────────────────────────

// indexArray of the second call restarts from 0 and is appended after the first's range.
TEST(AddSequenceSecond, IndexArray_AppendsFreshRange) {
	thrust::host_vector<int>  idx, id;
	thrust::host_vector<char> bases;
	addSequence(4, 3, "ACGT", idx, id, bases);  // appends [0..5]
	addSequence(3, 3, "TTG",  idx, id, bases);  // appends [0..4]
	EXPECT_EQ(toVec(idx), (std::vector<int>{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4}));
}

// IDArray for the second sequence uses IDArray.back() + 1 from the previous call.
TEST(AddSequenceSecond, IDArray_IncrementsPerSequence) {
	thrust::host_vector<int>  idx, id;
	thrust::host_vector<char> bases;
	addSequence(4, 3, "ACGT", idx, id, bases);  // fills with 0s (size 6)
	addSequence(3, 3, "TTG",  idx, id, bases);  // appends 1s (size 5)
	EXPECT_EQ(toVec(id), (std::vector<int>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1}));
}

// baseArray for the second call appends FASTAseq + overlap after the first's content.
TEST(AddSequenceSecond, BaseArray_AppendsSeqThenOverlap) {
	thrust::host_vector<int>  idx, id;
	thrust::host_vector<char> bases;
	addSequence(4, 3, "ACGT", idx, id, bases);  // "ACGT" + "AC"
	addSequence(3, 3, "TTG",  idx, id, bases);  // "TTG"  + "TT"
	EXPECT_EQ(toVec(bases),
		(std::vector<char>{'A', 'C', 'G', 'T', 'A', 'C', 'T', 'T', 'G', 'T', 'T'}));
}

// ─── third call ──────────────────────────────────────────────────────────────

// After three calls, ID increments to 2 and indexArray restarts from 0 each time.
TEST(AddSequenceThird, IDAndIndexArray) {
	thrust::host_vector<int>  idx, id;
	thrust::host_vector<char> bases;
	addSequence(2, 2, "AC", idx, id, bases);
	addSequence(2, 2, "GT", idx, id, bases);
	addSequence(2, 2, "TT", idx, id, bases);
	// jointLength = 1; each call appends 2 + 1 = 3 entries
	EXPECT_EQ(toVec(id),  (std::vector<int>{0, 0, 0, 1, 1, 1, 2, 2, 2}));
	EXPECT_EQ(toVec(idx), (std::vector<int>{0, 1, 2, 0, 1, 2, 0, 1, 2}));
}
