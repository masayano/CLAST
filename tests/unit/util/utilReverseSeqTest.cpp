#include "util/utilReverseSeq.hpp"

#include <gtest/gtest.h>

#include <string>

/* compBase: W–C pair for a single IUPAC base (unknown -> N). */
TEST(CompBase, WatsonCrick) {
	EXPECT_EQ(compBase('A'), 'T');
	EXPECT_EQ(compBase('T'), 'A');
	EXPECT_EQ(compBase('G'), 'C');
	EXPECT_EQ(compBase('C'), 'G');
}

TEST(CompBase, NonAtgcReturnsN) { EXPECT_EQ(compBase('n'), 'N'); }

/* compSeq: reverse-complement (Watson–Crick), one char at a time from 3' end. */
TEST(CompSeq, Empty) { EXPECT_EQ(compSeq(""), ""); }

TEST(CompSeq, SingleBases) {
	EXPECT_EQ(compSeq("A"), "T");
	EXPECT_EQ(compSeq("T"), "A");
	EXPECT_EQ(compSeq("G"), "C");
	EXPECT_EQ(compSeq("C"), "G");
}

TEST(CompSeq, UnknownBecomesN) {
	EXPECT_EQ(compSeq("N"), "N");
	EXPECT_EQ(compSeq("X"), "N");
}

TEST(CompSeq, ShortSequence) {
	EXPECT_EQ(compSeq("AT"), "AT");
	// "GC" -> reverse then complement -> "GC" (C,G -> G,C in output order = "GC")
	EXPECT_EQ(compSeq("GC"), "GC");
}

TEST(CompSeq, ForwardStrandExample) {
	EXPECT_EQ(compSeq("ATGC"), "GCAT");
}

// Lowercase letters are not ACGT uppercase → compBase returns 'N' for all of them.
TEST(CompBase, LowercaseReturnsN) {
	EXPECT_EQ(compBase('a'), 'N');
	EXPECT_EQ(compBase('t'), 'N');
	EXPECT_EQ(compBase('g'), 'N');
	EXPECT_EQ(compBase('c'), 'N');
}

// Other non-ACGT characters (uppercase and symbols) also return 'N'.
TEST(CompBase, OtherNonACGTReturnsN) {
	EXPECT_EQ(compBase('U'), 'N');
	EXPECT_EQ(compBase('B'), 'N');
	EXPECT_EQ(compBase('1'), 'N');
	EXPECT_EQ(compBase('-'), 'N');
}

// compSeq on a longer sequence: reverse-complement is applied correctly.
// "AATGCC" → reverse: "CCGTAA" → complement each: "GGCATT"
TEST(CompSeq, LongSequence) {
	EXPECT_EQ(compSeq("AATGCC"), "GGCATT");
}

// Lowercase in a sequence: each base becomes 'N', then reversed.
TEST(CompSeq, LowercaseBecomesAllN) {
	EXPECT_EQ(compSeq("atgc"), "NNNN");
}

// Sequence already containing only N stays all-N after reverse-complement.
TEST(CompSeq, AllNSequenceStaysAllN) {
	EXPECT_EQ(compSeq("NNN"), "NNN");
}

// Mixed uppercase ACGT and unknown: unknowns become 'N', ACGT complement normally.
// "ANTG" → reverse: "GTNA" → complement: "CANT"
TEST(CompSeq, MixedKnownAndUnknown) {
	EXPECT_EQ(compSeq("ANTG"), "CANT");
}
