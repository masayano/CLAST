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
