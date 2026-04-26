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
