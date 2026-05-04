// Unit tests for `src/util/utilReverseSeq.cpp` / `utilReverseSeq.hpp`:
// Watson–Crick complement (`compBase`) and reverse-complement (`compSeq`).

#include "util/utilReverseSeq.hpp"

#include <gtest/gtest.h>

#include <string>

// Strict ACGT pairing in switch; anything else falls through to default → 'N'.
TEST(CompBase, WatsonCrick) {
	EXPECT_EQ(compBase('A'), 'T');
	EXPECT_EQ(compBase('T'), 'A');
	EXPECT_EQ(compBase('G'), 'C');
	EXPECT_EQ(compBase('C'), 'G');
}

// Lowercase (and other non-switch cases) → default branch → 'N' (case-sensitive API).
TEST(CompBase, NonAtgcReturnsN) { EXPECT_EQ(compBase('n'), 'N'); }

// Empty input → empty output; reverse iteration has nothing to consume.
TEST(CompSeq, Empty) { EXPECT_EQ(compSeq(""), ""); }

// Single-base strings: reverse order trivial; complement maps each ACGT endpoint.
TEST(CompSeq, SingleBases) {
	EXPECT_EQ(compSeq("A"), "T");
	EXPECT_EQ(compSeq("T"), "A");
	EXPECT_EQ(compSeq("G"), "C");
	EXPECT_EQ(compSeq("C"), "G");
}

// Unknown monomers complement to N; palindromic unknown stays N.
TEST(CompSeq, UnknownBecomesN) {
	EXPECT_EQ(compSeq("N"), "N");
	EXPECT_EQ(compSeq("X"), "N");
}

// Length-2 palindromes under RC: order reverses then complements → same string ("AT", "GC").
TEST(CompSeq, ShortSequence) {
	EXPECT_EQ(compSeq("AT"), "AT");
	EXPECT_EQ(compSeq("GC"), "GC");
}

// Canonical four-mer check: reverse "CGTA" then complement → "GCAT".
TEST(CompSeq, ForwardStrandExample) {
	EXPECT_EQ(compSeq("ATGC"), "GCAT");
}

// Lowercase letters hit default in compBase → each becomes N; output order still reversed.
TEST(CompBase, LowercaseReturnsN) {
	EXPECT_EQ(compBase('a'), 'N');
	EXPECT_EQ(compBase('t'), 'N');
	EXPECT_EQ(compBase('g'), 'N');
	EXPECT_EQ(compBase('c'), 'N');
}

// Non-ACGT uppercase / symbols: same default → N per position.
TEST(CompBase, OtherNonACGTReturnsN) {
	EXPECT_EQ(compBase('U'), 'N');
	EXPECT_EQ(compBase('B'), 'N');
	EXPECT_EQ(compBase('1'), 'N');
	EXPECT_EQ(compBase('-'), 'N');
}

// Step-by-step: "AATGCC" reversed → "CCGTAA"; complement → "GGCATT".
TEST(CompSeq, LongSequence) {
	EXPECT_EQ(compSeq("AATGCC"), "GGCATT");
}

// Entire lowercase string → all N before reverse → still all N.
TEST(CompSeq, LowercaseBecomesAllN) {
	EXPECT_EQ(compSeq("atgc"), "NNNN");
}

// All-N input: reverse unchanged; compBase('N') → 'N'.
TEST(CompSeq, AllNSequenceStaysAllN) {
	EXPECT_EQ(compSeq("NNN"), "NNN");
}

// Mixed: known bases complement; unknown maps to N. "ANTG" → reverse "GTNA" → "CANT".
TEST(CompSeq, MixedKnownAndUnknown) {
	EXPECT_EQ(compSeq("ANTG"), "CANT");
}
