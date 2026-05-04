// Unit tests for `src/util/fastaDataSizeUtil.cpp` / `fastaDataSizeUtil.hpp`:
// `calcTotalDbSizeFromFastaPaths` — sums lengths of non-header FASTA lines across files.

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

// No paths → loop body never runs → total 0 (baseline for callers aggregating DB size).
TEST(FastaDataSize, EmptyPathList) {
	EXPECT_EQ(calcTotalDbSizeFromFastaPaths({}), 0L);
}

// Classic FASTA: lines containing `>` are skipped; only residue lines contribute `buf.size()`
// (newlines stripped by `getline`). Two sequences "ATGC" (4) + "GG" (2) → 6.
TEST(FastaDataSize, SumsDataLinesExcludingHeader) {
	const std::string name = "clast_ut_fasta_data_1.fa";
	const fs::path p = writeTempFasta(
			name, ">seq1\nATGC\n>seq2\nGG\n");
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	ASSERT_EQ(n, 4L + 2L);
}

// Rule is `buf.find('>') != npos` → skip entire line if `>` appears anywhere (not only column 0).
// "AT>GC" skipped; "XX" counts 2.
TEST(FastaDataSize, SkipsLineContainingAnyGreaterThan) {
	const std::string name = "clast_ut_fasta_data_2.fa";
	const fs::path p = writeTempFasta(
			name, "AT>GC\nXX\n");
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	EXPECT_EQ(n, 2L);
}

// Multiple paths: each file scanned independently; totals summed (order of paths irrelevant).
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

// Only headers → every line skipped → contribution 0 from this file.
TEST(FastaDataSize, FileWithOnlyHeadersIsZero) {
	const fs::path p = writeTempFasta(
			"clast_ut_fasta_headers_only.fa", ">seq1\n>seq2\n");
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	EXPECT_EQ(n, 0L);
}

// Empty stream: `getline` never yields data lines → 0.
TEST(FastaDataSize, EmptyFileIsZero) {
	const fs::path p = writeTempFasta("clast_ut_fasta_empty.fa", "");
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	EXPECT_EQ(n, 0L);
}

// Mid-line `>` skips whole line; following pure data line contributes full length ("GGG" → 3).
TEST(FastaDataSize, MidLineGreaterThanSkipsEntireLine) {
	const fs::path p = writeTempFasta(
			"clast_ut_fasta_midgt.fa", "ATGC>X\nGGG\n");
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	EXPECT_EQ(n, 3L);
}

// Blank lines (getline → empty `buf`) contain no `>` → counted as 0-length data lines.
// ">header\n\nAT\n\n" → 0 + 2 + 0 = 2.
TEST(FastaDataSize, BlankDataLinesContributeZero) {
	const fs::path p = writeTempFasta(
			"clast_ut_fasta_blank.fa", ">header\n\nAT\n\n");
	const long n = calcTotalDbSizeFromFastaPaths({p.string()});
	fs::remove(p);
	EXPECT_EQ(n, 0L + 2L + 0L);
}
