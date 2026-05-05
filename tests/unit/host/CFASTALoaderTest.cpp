#include "host/CFASTALoader.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

// ─── judgeContinueReading ────────────────────────────────────────────────────
//
// Behaviour summary:
//   label empty                                   → true,  nothing modified
//   label set, sequenceLength > availableRAMSize  → true,  sequence skipped;
//                                                           label/seq/startIdx cleared, readPos=pos
//   label set, sumReadSize+len < availableRAMSize → true,  entry added to FASTA;
//                                                           label/seq/startIdx cleared, sumReadSize+=len, readPos=pos
//   label set, sumReadSize+len >= availableRAMSize
//              (but len <= availableRAMSize)       → false, sumReadSize=0;
//                                                           label/seq/startIdx unchanged

// Empty label: function returns true and leaves all state untouched.
TEST(JudgeContinueReading, EmptyLabel_ReturnsTrueNoChanges) {
	std::string label, sequence = "ACGT";
	int startIdx = 3;
	std::vector<CHostFASTA> fasta;
	long sumReadSize = 5, readPos = 10;

	bool result = clast::fasta::judgeContinueReading(
			"target", /*pos=*/99, /*availableRAMSize=*/100,
			label, sequence, startIdx, fasta, sumReadSize, readPos);

	EXPECT_TRUE(result);
	EXPECT_EQ(fasta.size(), 0u);
	EXPECT_EQ(sequence,    "ACGT");
	EXPECT_EQ(sumReadSize, 5);
	EXPECT_EQ(readPos,     10);
}

// Sequence longer than availableRAMSize: skipped, state cleared, returns true.
TEST(JudgeContinueReading, SequenceTooLong_ClearsStateAndReturnsTrue) {
	std::string label = "seq1", sequence = "ACGT";  // length 4
	int startIdx = 2;
	std::vector<CHostFASTA> fasta;
	long sumReadSize = 0, readPos = 0;

	bool result = clast::fasta::judgeContinueReading(
			"target", /*pos=*/42, /*availableRAMSize=*/3,
			label, sequence, startIdx, fasta, sumReadSize, readPos);

	EXPECT_TRUE(result);
	EXPECT_TRUE(label.empty());
	EXPECT_TRUE(sequence.empty());
	EXPECT_EQ(startIdx,   0);
	EXPECT_EQ(readPos,    42);
	EXPECT_EQ(fasta.size(), 0u);
}

// Sequence fits: added to FASTA with correct label and sequence.
TEST(JudgeContinueReading, SequenceFits_AddsEntryAndReturnsTrue) {
	std::string label = "seq1", sequence = "ACGT";
	int startIdx = 0;
	std::vector<CHostFASTA> fasta;
	long sumReadSize = 0, readPos = 0;

	bool result = clast::fasta::judgeContinueReading(
			"target", /*pos=*/50, /*availableRAMSize=*/100,
			label, sequence, startIdx, fasta, sumReadSize, readPos);

	EXPECT_TRUE(result);
	ASSERT_EQ(fasta.size(), 1u);
	EXPECT_EQ(fasta[0].getLabel(),    "seq1");
	EXPECT_EQ(fasta[0].getSequence(), "ACGT");
}

// Sequence fits: startIdx at point of add is stored in the FASTA entry, then cleared.
TEST(JudgeContinueReading, SequenceFits_StoresAndClearsStartIdx) {
	std::string label = "seq1", sequence = "ACGT";
	int startIdx = 7;
	std::vector<CHostFASTA> fasta;
	long sumReadSize = 0, readPos = 0;

	clast::fasta::judgeContinueReading(
			"target", 0, 100, label, sequence, startIdx, fasta, sumReadSize, readPos);

	ASSERT_EQ(fasta.size(), 1u);
	EXPECT_EQ(fasta[0].getStartIdx(), 7);
	EXPECT_EQ(startIdx, 0);  // cleared after use
}

// Sequence fits: label/seq cleared, sumReadSize and readPos updated.
TEST(JudgeContinueReading, SequenceFits_UpdatesStateCorrectly) {
	std::string label = "seq1", sequence = "ACG";  // length 3
	int startIdx = 0;
	std::vector<CHostFASTA> fasta;
	long sumReadSize = 0, readPos = 0;

	clast::fasta::judgeContinueReading(
			"target", /*pos=*/123, 100, label, sequence, startIdx, fasta, sumReadSize, readPos);

	EXPECT_TRUE(label.empty());
	EXPECT_TRUE(sequence.empty());
	EXPECT_EQ(sumReadSize, 3);
	EXPECT_EQ(readPos,     123);
}

// Sequence fits twice: sumReadSize accumulates across consecutive calls.
TEST(JudgeContinueReading, SequenceFits_AccumulatesSumReadSize) {
	std::vector<CHostFASTA> fasta;
	long sumReadSize = 0, readPos = 0;

	std::string label1 = "s1", seq1 = "ACG";   // length 3
	int idx1 = 0;
	clast::fasta::judgeContinueReading("target", 10, 100, label1, seq1, idx1, fasta, sumReadSize, readPos);

	std::string label2 = "s2", seq2 = "TTTT";  // length 4
	int idx2 = 0;
	clast::fasta::judgeContinueReading("target", 20, 100, label2, seq2, idx2, fasta, sumReadSize, readPos);

	EXPECT_EQ(fasta.size(), 2u);
	EXPECT_EQ(sumReadSize,  7);
	EXPECT_EQ(readPos,      20);
}

// Buffer full: returns false, sumReadSize reset to 0; label/seq untouched.
TEST(JudgeContinueReading, BufferFull_ReturnsFalseAndResetsSumReadSize) {
	std::string label = "seq1", sequence = "ACGT";  // length 4
	int startIdx = 0;
	std::vector<CHostFASTA> fasta;
	long sumReadSize = 8, readPos = 0;  // 8+4=12 >= availableRAMSize(10)

	bool result = clast::fasta::judgeContinueReading(
			"target", 0, /*availableRAMSize=*/10,
			label, sequence, startIdx, fasta, sumReadSize, readPos);

	EXPECT_FALSE(result);
	EXPECT_EQ(sumReadSize,   0);
	EXPECT_EQ(fasta.size(), 0u);
	EXPECT_EQ(label,    "seq1");   // not cleared on buffer-full
	EXPECT_EQ(sequence, "ACGT");   // not cleared on buffer-full
}

// Sequence length exactly equals availableRAMSize: treated as buffer full (strict <).
TEST(JudgeContinueReading, SequenceExactlyFillsRAM_ReturnsFalse) {
	std::string label = "seq1", sequence = "ACGT";  // length 4
	int startIdx = 0;
	std::vector<CHostFASTA> fasta;
	long sumReadSize = 0, readPos = 0;

	bool result = clast::fasta::judgeContinueReading(
			"target", 0, /*availableRAMSize=*/4,
			label, sequence, startIdx, fasta, sumReadSize, readPos);

	// 4 > 4? No. 0+4 < 4? No → buffer full.
	EXPECT_FALSE(result);
	EXPECT_EQ(fasta.size(), 0u);
}
