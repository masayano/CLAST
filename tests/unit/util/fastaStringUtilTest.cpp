// Unit tests for `src/util/fastaStringUtil.cpp` / `fastaStringUtil.hpp`:
// `removeSpace` (collapse ASCII spaces) and `replaceSmallToBig` (normalize to A–Z / N).

#include "util/fastaStringUtil.hpp"

#include <gtest/gtest.h>

#include <string>

// Empty string: loop never replaces → remains empty.
TEST(RemoveSpace, EmptyUnchanged) {
	std::string s;
	removeSpace(s);
	EXPECT_EQ(s, "");
}

// Every literal space character removed; consecutive spaces each stripped in separate iterations.
TEST(RemoveSpace, RemovesAllSpaces) {
	std::string s = "a b  c";
	removeSpace(s);
	EXPECT_EQ(s, "abc");
}

// No spaces → find returns npos immediately → identity.
TEST(RemoveSpace, NoSpacesUnchanged) {
	std::string s = "ATGC";
	removeSpace(s);
	EXPECT_EQ(s, "ATGC");
}

// Already uppercase ACGT: neither branch (`> 'Z'` then `< 'A' || > 'Z'`) triggers replacement → unchanged.
TEST(ReplaceSmallToBig, UppercaseUnchangedIfACGT) {
	std::string s = "ATGC";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "ATGC");
}

// Lowercase a–z: first pass maps ASCII +32 segment to uppercase via `-32`; second pass keeps ['A','Z'].
TEST(ReplaceSmallToBig, LowercaseToUpperACGT) {
	std::string s = "atgc";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "ATGC");
}

// Digit example: not > 'Z' → no -32; `< 'A'` → each char becomes 'N'. Letters above 'a' → upper then kept.
// '12ab' → 'NNAB'.
TEST(ReplaceSmallToBig, NonAlphabeticOrAfterMappingToN) {
	std::string s = "12ab";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "NNAB");
}

// All 26 lowercase letters: `> 'Z'` true → -32 yields A–Z; second check passes → full alphabet preserved.
TEST(ReplaceSmallToBig, AllLowercaseLettersMappedToUppercase) {
	std::string s = "abcdefghijklmnopqrstuvwxyz";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
}

// Uppercase outside ACGT but inside A–Z (e.g. IUPAC ambiguity codes): stay uppercase, not mapped to N.
TEST(ReplaceSmallToBig, NonACGTUppercaseKept) {
	std::string s = "BDENRYZ";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "BDENRYZ");
}

// Digits: ASCII < 'A', not > 'Z' → immediate 'N' assignment for each.
TEST(ReplaceSmallToBig, DigitsToN) {
	std::string s = "0189";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "NNNN");
}

// ASCII 91–96 etc.: `> 'Z'` → -32 shifts into punctuation below 'A' → second predicate → 'N'.
// '[' (91) → 59 (';') → N.
TEST(ReplaceSmallToBig, SymbolsAboveZToN) {
	std::string s = "[`";  // ASCII 91, 96
	replaceSmallToBig(s);
	EXPECT_EQ(s, "NN");
}

// Whitespace tab: not > 'Z', < 'A' → 'N'.
TEST(ReplaceSmallToBig, TabToN) {
	std::string s = "\t";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "N");
}

// Zero-length buffer: loop range empty → unchanged.
TEST(ReplaceSmallToBig, EmptyUnchanged) {
	std::string s;
	replaceSmallToBig(s);
	EXPECT_EQ(s, "");
}
