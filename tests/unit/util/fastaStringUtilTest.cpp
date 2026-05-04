#include "util/fastaStringUtil.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(RemoveSpace, EmptyUnchanged) {
	std::string s;
	removeSpace(s);
	EXPECT_EQ(s, "");
}

TEST(RemoveSpace, RemovesAllSpaces) {
	std::string s = "a b  c";
	removeSpace(s);
	EXPECT_EQ(s, "abc");
}

TEST(RemoveSpace, NoSpacesUnchanged) {
	std::string s = "ATGC";
	removeSpace(s);
	EXPECT_EQ(s, "ATGC");
}

TEST(ReplaceSmallToBig, UppercaseUnchangedIfACGT) {
	std::string s = "ATGC";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "ATGC");
}

TEST(ReplaceSmallToBig, LowercaseToUpperACGT) {
	std::string s = "atgc";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "ATGC");
}

TEST(ReplaceSmallToBig, NonAlphabeticOrAfterMappingToN) {
	std::string s = "12ab";
	replaceSmallToBig(s);
	// '1','2' -> not > 'Z', so no -32; then < 'A' -> 'N'
	// 'a','b' -> to 'A','B' then stays
	EXPECT_EQ(s, "NNAB");
}

// All 26 lowercase letters: a-z → A-Z via -32. Then only non-['A','Z'] become N.
// All of a-z map to A-Z which are within ['A','Z'], so all survive unchanged after uppercasing.
TEST(ReplaceSmallToBig, AllLowercaseLettersMappedToUppercase) {
	std::string s = "abcdefghijklmnopqrstuvwxyz";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
}

// Non-ACGT uppercase letters (B, D, E, ...) are in ['A','Z'] → kept as-is, not converted to N.
TEST(ReplaceSmallToBig, NonACGTUppercaseKept) {
	std::string s = "BDENRYZ";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "BDENRYZ");
}

// Digits are < 'A' (ASCII < 65) → become N. No -32 step (they are not > 'Z').
TEST(ReplaceSmallToBig, DigitsToN) {
	std::string s = "0189";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "NNNN");
}

// Characters just above 'Z' (ASCII 91-96: [ \ ] ^ _ `) are > 'Z' → -32 applied first,
// then they land at ASCII 59-64 which are < 'A' → become N.
// Example: '[' = 91 → 91-32=59 (';') → < 65 → 'N'
TEST(ReplaceSmallToBig, SymbolsAboveZToN) {
	std::string s = "[`";  // ASCII 91, 96
	replaceSmallToBig(s);
	EXPECT_EQ(s, "NN");
}

// Tab ('\t' = ASCII 9) is < 'A' and not > 'Z' → directly becomes N.
TEST(ReplaceSmallToBig, TabToN) {
	std::string s = "\t";
	replaceSmallToBig(s);
	EXPECT_EQ(s, "N");
}

// Empty string is unchanged.
TEST(ReplaceSmallToBig, EmptyUnchanged) {
	std::string s;
	replaceSmallToBig(s);
	EXPECT_EQ(s, "");
}
