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
