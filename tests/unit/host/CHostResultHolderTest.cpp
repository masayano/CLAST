// Unit tests for `clast::result::withinOutputLimit` and
// `clast::result::printResultToFile` in `src/host/CHostResultHolder.cuh`.

#include "host/CHostResultHolder.cuh"

#include <gtest/gtest.h>

#include <thrust/host_vector.h>
#include <fstream>
#include <string>
#include <vector>

using namespace clast::result;

// ─── withinOutputLimit ───────────────────────────────────────────────────────

// numberOfOutput == -1: always true regardless of count or query label.
TEST(WithinOutputLimitTest, Unlimited_AlwaysTrue) {
	EXPECT_TRUE(withinOutputLimit(-1, "A", "A", 100));
	EXPECT_TRUE(withinOutputLimit(-1, "A", "B", 100));
}

// Different query labels: first hit of a new query is always written.
TEST(WithinOutputLimitTest, NewQuery_AlwaysTrue) {
	EXPECT_TRUE(withinOutputLimit(1, "A", "B", 5));
}

// Same query, count is still under the limit.
TEST(WithinOutputLimitTest, SameQuery_CountUnderLimit_True) {
	EXPECT_TRUE(withinOutputLimit(3, "A", "A", 2));
}

// Same query, count exactly equals the limit → no more hits allowed.
TEST(WithinOutputLimitTest, SameQuery_CountAtLimit_False) {
	EXPECT_FALSE(withinOutputLimit(3, "A", "A", 3));
}

// Same query, count exceeds the limit.
TEST(WithinOutputLimitTest, SameQuery_CountOverLimit_False) {
	EXPECT_FALSE(withinOutputLimit(1, "A", "A", 5));
}

// ─── fixture ─────────────────────────────────────────────────────────────────

class PrintResultToFileTest : public ::testing::Test {
protected:
	std::string path;

	void SetUp() override {
		std::string dir = ::testing::TempDir();
		if(dir.back() != '/') dir += '/';
		path = dir + "clast_prtf_test.tsv";
		std::remove(path.c_str());
	}

	void TearDown() override {
		std::remove(path.c_str());
	}

	std::string readOutput() {
		std::ifstream ifs(path);
		return {std::istreambuf_iterator<char>(ifs), {}};
	}

	// Convenience wrapper with minimal arrays for single-hit scenarios.
	// queryID 0 → queryLabelArray[0] / queryStrandArray[0]
	// targetID 0 → targetLabelArray[0] / targetStartIdxArray[0]
	void runSingle(
			int numOut,
			const std::string& queryLabel,
			char queryStrand,
			const std::string& targetLabel,
			int targetStartIdx,
			int tID, int tIdx, int qID, int qIdx,
			int tHitLen, int qHitLen, int matchNum, int score, double evalue) {
		printResultToFile(
				numOut, path,
				{queryLabel, queryLabel},
				{'+', '-'},
				{targetLabel},
				{targetStartIdx},
				{tID}, {tIdx}, {qID}, {qIdx},
				{tHitLen}, {qHitLen}, {matchNum}, {score}, {evalue});
	}
};

// ─── empty input ─────────────────────────────────────────────────────────────

// No hits → output file is created but contains no lines.
TEST_F(PrintResultToFileTest, EmptyHits_FileIsEmpty) {
	printResultToFile(
			-1, path,
			{}, {}, {}, {},
			{}, {}, {}, {},
			{}, {}, {}, {}, {});
	EXPECT_EQ(readOutput(), "");
}

// ─── TSV field format ────────────────────────────────────────────────────────

// Single hit: verify all ten tab-separated fields are written correctly.
// Format: queryLabel \t queryIndex \t qHitLength \t queryStrand \t
//         targetLabel \t targetIndex \t tHitLength \t
//         matchNum(matchNum*100/qHitLength%) \t score \t evalue
TEST_F(PrintResultToFileTest, SingleHit_FieldsCorrect) {
	// qHitLength=4, matchNum=3 → 3(75%)
	// targetIndex = tIdx(5) + targetStartIdx(100) = 105
	runSingle(-1, "QueryA", '+', "TargetX", 100, 0, 5, 0, 10, 20, 4, 3, 50, 0.0);
	EXPECT_EQ(readOutput(), "QueryA\t10\t4\t+\tTargetX\t105\t20\t3(75%)\t50\t0\n");
}

// Negative strand hit: queryStrand field is '-'.
TEST_F(PrintResultToFileTest, SingleHit_NegativeStrand) {
	// queryID=1 → strand '-'
	runSingle(-1, "QueryA", '-', "TargetX", 0, 0, 0, 1, 0, 10, 2, 2, 10, 0.0);
	EXPECT_EQ(readOutput(), "QueryA\t0\t2\t-\tTargetX\t0\t10\t2(100%)\t10\t0\n");
}

// targetIndex incorporates the per-target start offset.
TEST_F(PrintResultToFileTest, TargetIndexIncludesStartOffset) {
	runSingle(-1, "Q", '+', "T", 200, 0, 7, 0, 0, 10, 10, 10, 0, 0.0);
	// targetIndex = 7 + 200 = 207
	EXPECT_EQ(readOutput(), "Q\t0\t10\t+\tT\t207\t10\t10(100%)\t0\t0\n");
}

// ─── numberOfOutput limit ────────────────────────────────────────────────────

// numberOfOutput = -1: all hits for a query are written.
TEST_F(PrintResultToFileTest, Unlimited_AllHitsWritten) {
	printResultToFile(
			-1, path,
			{"Q", "Q"}, {'+', '-'},
			{"T"}, {0},
			{0, 0, 0},   // tID
			{0, 1, 2},   // tIdx
			{0, 0, 0},   // qID
			{0, 0, 0},   // qIdx
			{10, 10, 10}, {4, 4, 4}, {4, 4, 4}, {30, 20, 10}, {0.0, 0.0, 0.0});
	const std::string out = readOutput();
	EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 3);
}

// numberOfOutput = 1: only the first (top) hit per query is written.
TEST_F(PrintResultToFileTest, LimitOne_OnlyTopHitWritten) {
	printResultToFile(
			1, path,
			{"Q", "Q"}, {'+', '-'},
			{"T"}, {0},
			{0, 0, 0},   // tID
			{0, 1, 2},   // tIdx  (distinct → dedup passes)
			{0, 0, 0},   // qID
			{0, 0, 0},   // qIdx
			{10, 10, 10}, {4, 4, 4}, {4, 4, 4}, {30, 20, 10}, {0.0, 0.0, 0.0});
	const std::string out = readOutput();
	EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 1);
}

// numberOfOutput = 2: first two hits written, third skipped.
TEST_F(PrintResultToFileTest, LimitTwo_FirstTwoHitsWritten) {
	printResultToFile(
			2, path,
			{"Q", "Q"}, {'+', '-'},
			{"T"}, {0},
			{0, 0, 0},
			{0, 1, 2},
			{0, 0, 0},
			{0, 0, 0},
			{10, 10, 10}, {4, 4, 4}, {4, 4, 4}, {30, 20, 10}, {0.0, 0.0, 0.0});
	const std::string out = readOutput();
	EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 2);
}

// The first hit of each new query is always written regardless of the limit.
TEST_F(PrintResultToFileTest, LimitOne_TopHitOfEachQueryWritten) {
	// Two queries (Q0/Q1) each with 2 hits; limit=1 → 2 lines total
	printResultToFile(
			1, path,
			{"Q0", "Q0", "Q1", "Q1"}, {'+', '-', '+', '-'},
			{"T"}, {0},
			{0, 0, 0, 0},   // tID
			{0, 1, 2, 3},   // tIdx
			{0, 0, 2, 2},   // qID (0→Q0+, 2→Q1+)
			{0, 0, 0, 0},
			{10,10,10,10}, {4,4,4,4}, {4,4,4,4}, {30,20,30,20}, {0.0,0.0,0.0,0.0});
	const std::string out = readOutput();
	EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 2);
}

// ─── deduplication ───────────────────────────────────────────────────────────

// Consecutive rows with identical fields → only the first is written.
TEST_F(PrintResultToFileTest, DuplicateRow_OnlyFirstWritten) {
	// Two hits with identical field values → dedup suppresses second
	printResultToFile(
			-1, path,
			{"Q", "Q"}, {'+', '-'},
			{"T"}, {0},
			{0, 0},
			{5, 5},   // same tIdx
			{0, 0},
			{3, 3},   // same qIdx
			{10, 10}, {4, 4}, {3, 3}, {50, 50}, {0.0, 0.0});
	const std::string out = readOutput();
	EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 1);
}

// Rows differing in any field pass the dedup check.
TEST_F(PrintResultToFileTest, RowDifferingInOneField_BothWritten) {
	// Two hits identical except score
	printResultToFile(
			-1, path,
			{"Q", "Q"}, {'+', '-'},
			{"T"}, {0},
			{0, 0},
			{5, 5},
			{0, 0},
			{3, 3},
			{10, 10}, {4, 4}, {3, 3}, {50, 40}, {0.0, 0.0});
	const std::string out = readOutput();
	EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 2);
}

// ─── append mode ─────────────────────────────────────────────────────────────

// A second call appends to the existing file rather than overwriting.
TEST_F(PrintResultToFileTest, SecondCall_AppendsToFile) {
	runSingle(-1, "Q", '+', "T", 0, 0, 0, 0, 0, 10, 4, 4, 10, 0.0);
	runSingle(-1, "Q", '+', "T", 0, 0, 1, 0, 0, 10, 4, 4, 10, 0.0);
	const std::string out = readOutput();
	EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 2);
}
