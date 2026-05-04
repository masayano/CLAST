#include "util/fastaDataSizeUtil.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

// Portable temp file under the system temp directory (C++17).
fs::path writeTempFasta(const std::string& name, const std::string& content) {
	const fs::path p = fs::temp_directory_path() / name;
	{
		std::ofstream f(p, std::ios::binary);
		f << content;
	}
	return p;
}

} // namespace

TEST(FastaDataSize, EmptyPathList) {
	EXPECT_EQ(calcTotalDbSizeFromFastaPaths({}), 0L);
}

TEST(FastaDataSize, SumsDataLinesExcludingHeader) {
	const std::string name = "clast_ut_fasta_data_1.fa";
	const fs::path p = writeTempFasta(
			name, ">seq1\nATGC\n>seq2\nGG\n");
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	ASSERT_EQ(n, 4L + 2L);
}

TEST(FastaDataSize, SkipsLineContainingAnyGreaterThan) {
	// Per implementation: a line is skipped if it contains '>'.
	const std::string name = "clast_ut_fasta_data_2.fa";
	const fs::path p = writeTempFasta(
			name, "AT>GC\nXX\n");
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	EXPECT_EQ(n, 2L);
}

TEST(FastaDataSize, MultipleFiles) {
	const std::string n1 = "clast_ut_fasta_m1.fa";
	const std::string n2 = "clast_ut_fasta_m2.fa";
	const fs::path p1 = writeTempFasta(n1, ">a\nA\n");
	const fs::path p2 = writeTempFasta(n2, "TT\n");
	const long n = calcTotalDbSizeFromFastaPaths(
			{ p1.string(), p2.string() });
	fs::remove(p1);
	fs::remove(p2);
	EXPECT_EQ(n, 1L + 2L);
}

// File that contains only header lines contributes 0.
TEST(FastaDataSize, FileWithOnlyHeadersIsZero) {
	const fs::path p = writeTempFasta(
			"clast_ut_fasta_headers_only.fa", ">seq1\n>seq2\n");
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	EXPECT_EQ(n, 0L);
}

// A completely empty file contributes 0.
TEST(FastaDataSize, EmptyFileIsZero) {
	const fs::path p = writeTempFasta("clast_ut_fasta_empty.fa", "");
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	EXPECT_EQ(n, 0L);
}

// '>' anywhere in the line (not just at position 0) causes the whole line to be skipped.
// The SkipsLineContainingAnyGreaterThan test already covers mid-line '>'.
// This test verifies the retained data line contributes its full length.
TEST(FastaDataSize, MidLineGreaterThanSkipsEntireLine) {
	const fs::path p = writeTempFasta(
			"clast_ut_fasta_midgt.fa", "ATGC>X\nGGG\n");
	// First line has '>' → skipped (0 chars). Second line: "GGG\n" → 3 chars.
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	EXPECT_EQ(n, 3L);
}

// Newline-only lines (empty data lines) contribute 0 each.
TEST(FastaDataSize, BlankDataLinesContributeZero) {
	const fs::path p = writeTempFasta(
			"clast_ut_fasta_blank.fa", ">header\n\nAT\n\n");
	// "\n" line: buf="" (getline strips newline), no '>'; buf.size()=0
	// "AT\n" line: buf="AT", size=2
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	EXPECT_EQ(n, 0L + 2L + 0L);
}
