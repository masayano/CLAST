// Unit tests for `src/cli/listenCommandLine.cpp` / `listenCommandLine.hpp`:
// state-machine command-line parser for CLAST.

#include "cli/listenCommandLine.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

// ─── helpers ────────────────────────────────────────────────────────────────

struct Params {
	double tRAM = 0, qRAM = 0, tVRAM = 0, qVRAM = 0, cutOff = 0;
	int lMer = 0, stride = 0, repeat = 0, width = 0, gap = 0,
	    numOut = 0, local = 0, device = 0, sleep = 0;
	std::vector<std::string> tFiles, qFiles;
	std::string oFile;
};

static Params parse(std::initializer_list<const char*> args) {
	std::vector<const char*> argv(args);
	Params p;
	listenCommandLine(
			argv.size(), argv.data(),
			p.tRAM, p.qRAM, p.tVRAM, p.qVRAM,
			p.lMer, p.stride, p.repeat, p.width, p.gap,
			p.numOut, p.local, p.device, p.sleep, p.cutOff,
			p.tFiles, p.qFiles, p.oFile);
	return p;
}

// Wrapper for death tests: avoids repeating all output parameters inline.
#define PARSE_DEATH(argv_init_list) \
	do { \
		Params _p; \
		std::vector<const char*> _av argv_init_list; \
		EXPECT_DEATH(listenCommandLine( \
				_av.size(), _av.data(), \
				_p.tRAM, _p.qRAM, _p.tVRAM, _p.qVRAM, \
				_p.lMer, _p.stride, _p.repeat, _p.width, _p.gap, \
				_p.numOut, _p.local, _p.device, _p.sleep, _p.cutOff, \
				_p.tFiles, _p.qFiles, _p.oFile), ".*"); \
	} while(false)

// ─── double parameters ───────────────────────────────────────────────────────

TEST(ListenCommandLine, ParsesTargetRAM) {
	EXPECT_DOUBLE_EQ(parse({"prog", "-tRAM", "1.5"}).tRAM, 1.5);
}

TEST(ListenCommandLine, ParsesQueryRAM) {
	EXPECT_DOUBLE_EQ(parse({"prog", "-qRAM", "2.5"}).qRAM, 2.5);
}

TEST(ListenCommandLine, ParsesTargetVRAM) {
	EXPECT_DOUBLE_EQ(parse({"prog", "-tVRAM", "3.5"}).tVRAM, 3.5);
}

TEST(ListenCommandLine, ParsesQueryVRAM) {
	EXPECT_DOUBLE_EQ(parse({"prog", "-qVRAM", "4.5"}).qVRAM, 4.5);
}

TEST(ListenCommandLine, ParsesCutOff) {
	EXPECT_DOUBLE_EQ(parse({"prog", "-cutOff", "0.001"}).cutOff, 0.001);
}

// ─── int parameters ──────────────────────────────────────────────────────────

TEST(ListenCommandLine, ParsesLMer) {
	EXPECT_EQ(parse({"prog", "-lMer", "11"}).lMer, 11);
}

TEST(ListenCommandLine, ParsesStride) {
	EXPECT_EQ(parse({"prog", "-stride", "3"}).stride, 3);
}

TEST(ListenCommandLine, ParsesRepeat) {
	EXPECT_EQ(parse({"prog", "-repeat", "5"}).repeat, 5);
}

TEST(ListenCommandLine, ParsesWidth) {
	EXPECT_EQ(parse({"prog", "-width", "10"}).width, 10);
}

TEST(ListenCommandLine, ParsesGap) {
	EXPECT_EQ(parse({"prog", "-gap", "2"}).gap, 2);
}

TEST(ListenCommandLine, ParsesNumOut) {
	EXPECT_EQ(parse({"prog", "-numOut", "100"}).numOut, 100);
}

TEST(ListenCommandLine, ParsesLocal) {
	EXPECT_EQ(parse({"prog", "-local", "1"}).local, 1);
}

TEST(ListenCommandLine, ParsesDevice) {
	EXPECT_EQ(parse({"prog", "-device", "2"}).device, 2);
}

TEST(ListenCommandLine, ParsesSleep) {
	EXPECT_EQ(parse({"prog", "-sleep", "30"}).sleep, 30);
}

// ─── file parameters ─────────────────────────────────────────────────────────

// -t collects filenames until the next flag.
TEST(ListenCommandLine, ParsesSingleTargetFile) {
	auto p = parse({"prog", "-t", "target.fa"});
	EXPECT_EQ(p.tFiles, (std::vector<std::string>{"target.fa"}));
}

TEST(ListenCommandLine, ParsesMultipleTargetFiles) {
	auto p = parse({"prog", "-t", "a.fa", "b.fa", "c.fa"});
	EXPECT_EQ(p.tFiles, (std::vector<std::string>{"a.fa", "b.fa", "c.fa"}));
}

// -q behaves the same as -t.
TEST(ListenCommandLine, ParsesQueryFiles) {
	auto p = parse({"prog", "-q", "q1.fa", "q2.fa"});
	EXPECT_EQ(p.qFiles, (std::vector<std::string>{"q1.fa", "q2.fa"}));
}

// -o takes exactly one filename and returns to START.
TEST(ListenCommandLine, ParsesOutputFile) {
	EXPECT_EQ(parse({"prog", "-o", "result.tsv"}).oFile, "result.tsv");
}

// ─── state transitions ───────────────────────────────────────────────────────

// A following flag ends -t file collection and switches state.
TEST(ListenCommandLine, TargetFilesEndAtNextFlag) {
	auto p = parse({"prog", "-t", "a.fa", "b.fa", "-lMer", "11"});
	EXPECT_EQ(p.tFiles, (std::vector<std::string>{"a.fa", "b.fa"}));
	EXPECT_EQ(p.lMer, 11);
}

// -t and -q can be interleaved with other parameters in any order.
TEST(ListenCommandLine, FullCommandLine) {
	auto p = parse({
			"prog",
			"-lMer",   "11",
			"-stride", "3",
			"-tRAM",   "8.0",
			"-t",      "target.fa",
			"-q",      "query.fa",
			"-o",      "out.tsv"});
	EXPECT_EQ(p.lMer, 11);
	EXPECT_EQ(p.stride, 3);
	EXPECT_DOUBLE_EQ(p.tRAM, 8.0);
	EXPECT_EQ(p.tFiles, (std::vector<std::string>{"target.fa"}));
	EXPECT_EQ(p.qFiles, (std::vector<std::string>{"query.fa"}));
	EXPECT_EQ(p.oFile, "out.tsv");
}

// ─── error cases ─────────────────────────────────────────────────────────────

// Unknown flag in START state → abort.
TEST(ListenCommandLine, UnknownFlag_Aborts) {
	PARSE_DEATH({"prog", "-unknown"});
}

// Non-numeric value for an int parameter → abort.
TEST(ListenCommandLine, BadIntValue_Aborts) {
	PARSE_DEATH({"prog", "-lMer", "notanumber"});
}

// Non-numeric value for a double parameter → abort.
TEST(ListenCommandLine, BadDoubleValue_Aborts) {
	PARSE_DEATH({"prog", "-tRAM", "notanumber"});
}

// Repeated -t while already collecting target files → abort.
TEST(ListenCommandLine, DuplicateTargetFlag_Aborts) {
	PARSE_DEATH({"prog", "-t", "a.fa", "-t"});
}

// Repeated -q while already collecting query files → abort.
TEST(ListenCommandLine, DuplicateQueryFlag_Aborts) {
	PARSE_DEATH({"prog", "-q", "a.fa", "-q"});
}
