// Unit tests for the clast::hash free functions declared in
// src/device/CDeviceHashTable.cuh:
//   createHashIndex, removeRepeat, makeIndexArray, makeGatewayKey,
//   makeCellSizeArray, makeGatewayIndex, removeRepeatedSequence.
//
// createHashIndex (complex CUDA kernel pipeline) is intentionally excluded.

#include "device/CDeviceHashTable.cuh"
#include "hitTestUtil.hpp"

#include <gtest/gtest.h>

using clast::test::hit::MakeIntVec;
using clast::test::hit::MakeLongVec;

// ─── removeRepeat ────────────────────────────────────────────────────────────

// cellSize strictly less than cutRepeat → kept as-is.
TEST(RemoveRepeat, KeepsCellSizeBelowCutoff) {
	clast::hash::removeRepeat f;
	EXPECT_EQ(f(5, 10), 5);
	EXPECT_EQ(f(1, 10), 1);
}

// cellSize equal to cutRepeat → zeroed (condition is strict <).
TEST(RemoveRepeat, ZerosCellSizeEqualToCutoff) {
	clast::hash::removeRepeat f;
	EXPECT_EQ(f(10, 10), 0);
}

// cellSize above cutRepeat → zeroed.
TEST(RemoveRepeat, ZerosCellSizeAboveCutoff) {
	clast::hash::removeRepeat f;
	EXPECT_EQ(f(15, 10), 0);
	EXPECT_EQ(f(100, 1), 0);
}

// ─── makeIndexArray ──────────────────────────────────────────────────────────

// Empty: baseLength=0 → size 0, no crash.
TEST(MakeIndexArray, EmptyWhenBaseLengthZero) {
	auto v = MakeIntVec({});
	clast::hash::makeIndexArray(0, 4, v);
	ASSERT_EQ(v.size(), 0u);
}

// Stride of 1: result is 0,1,2,...,n-1 scaled by 1 = 0,1,...
TEST(MakeIndexArray, StrideOneProducesConsecutiveIndices) {
	auto v = MakeIntVec({});
	clast::hash::makeIndexArray(4, 1, v);
	ASSERT_EQ(v.size(), 4u);
	EXPECT_EQ(v[0], 0); EXPECT_EQ(v[1], 1);
	EXPECT_EQ(v[2], 2); EXPECT_EQ(v[3], 3);
}

// General case: stride=4, baseLength=12 → {0, 4, 8}.
TEST(MakeIndexArray, ProducesStrideMultiples) {
	auto v = MakeIntVec({});
	clast::hash::makeIndexArray(12, 4, v);
	ASSERT_EQ(v.size(), 3u);
	EXPECT_EQ(v[0], 0);
	EXPECT_EQ(v[1], 4);
	EXPECT_EQ(v[2], 8);
}

// Integer division: baseLength=9, stride=4 → size 2, values {0, 4}.
TEST(MakeIndexArray, TruncatesWhenNotEvenleDivisible) {
	auto v = MakeIntVec({});
	clast::hash::makeIndexArray(9, 4, v);
	ASSERT_EQ(v.size(), 2u);
	EXPECT_EQ(v[0], 0);
	EXPECT_EQ(v[1], 4);
}

// ─── makeGatewayKey ──────────────────────────────────────────────────────────

// Empty input → empty gateway key.
TEST(MakeGatewayKey, EmptyInputProducesEmptyKey) {
	auto hashIndex  = MakeLongVec({});
	auto gatewayKey = MakeLongVec({});
	clast::hash::makeGatewayKey(hashIndex, gatewayKey);
	ASSERT_EQ(gatewayKey.size(), 0u);
}

// Single key: one unique value.
TEST(MakeGatewayKey, SingleKeyPassesThrough) {
	auto hashIndex  = MakeLongVec({7L});
	auto gatewayKey = MakeLongVec({});
	clast::hash::makeGatewayKey(hashIndex, gatewayKey);
	ASSERT_EQ(gatewayKey.size(), 1u);
	EXPECT_EQ(gatewayKey[0], 7L);
}

// All same: collapses to one unique key.
TEST(MakeGatewayKey, AllSameCollapsesToOne) {
	auto hashIndex  = MakeLongVec({3L, 3L, 3L, 3L});
	auto gatewayKey = MakeLongVec({});
	clast::hash::makeGatewayKey(hashIndex, gatewayKey);
	ASSERT_EQ(gatewayKey.size(), 1u);
	EXPECT_EQ(gatewayKey[0], 3L);
}

// Multiple distinct runs: each unique value kept in order.
TEST(MakeGatewayKey, ExtractsUniqueKeysFromSortedInput) {
	auto hashIndex  = MakeLongVec({1L, 1L, 2L, 3L, 3L, 3L});
	auto gatewayKey = MakeLongVec({});
	clast::hash::makeGatewayKey(hashIndex, gatewayKey);
	ASSERT_EQ(gatewayKey.size(), 3u);
	EXPECT_EQ(gatewayKey[0], 1L);
	EXPECT_EQ(gatewayKey[1], 2L);
	EXPECT_EQ(gatewayKey[2], 3L);
}

// ─── makeCellSizeArray ───────────────────────────────────────────────────────

// Empty input → empty output.
TEST(MakeCellSizeArray, EmptyInputProducesEmptyOutput) {
	auto hashIndex   = MakeLongVec({});
	auto gatewayKey  = MakeLongVec({});
	auto cellSizeArr = MakeIntVec({});
	clast::hash::makeCellSizeArray(hashIndex, gatewayKey, cellSizeArr);
	ASSERT_EQ(cellSizeArr.size(), 0u);
}

// One key, one entry → count=1.
TEST(MakeCellSizeArray, SingleEntryCountIsOne) {
	auto hashIndex   = MakeLongVec({5L});
	auto gatewayKey  = MakeLongVec({5L});
	auto cellSizeArr = MakeIntVec({});
	clast::hash::makeCellSizeArray(hashIndex, gatewayKey, cellSizeArr);
	ASSERT_EQ(cellSizeArr.size(), 1u);
	EXPECT_EQ(cellSizeArr[0], 1);
}

// All same key → one bucket with full count.
TEST(MakeCellSizeArray, AllSameKeyCountsAll) {
	auto hashIndex   = MakeLongVec({2L, 2L, 2L});
	auto gatewayKey  = MakeLongVec({2L});
	auto cellSizeArr = MakeIntVec({});
	clast::hash::makeCellSizeArray(hashIndex, gatewayKey, cellSizeArr);
	ASSERT_EQ(cellSizeArr.size(), 1u);
	EXPECT_EQ(cellSizeArr[0], 3);
}

// Multiple keys with distinct run lengths.
TEST(MakeCellSizeArray, MultipleBucketsWithCorrectCounts) {
	// hashIndex: 1×2, 2×1, 3×3  (sorted)
	auto hashIndex   = MakeLongVec({1L, 1L, 2L, 3L, 3L, 3L});
	auto gatewayKey  = MakeLongVec({1L, 2L, 3L});
	auto cellSizeArr = MakeIntVec({});
	clast::hash::makeCellSizeArray(hashIndex, gatewayKey, cellSizeArr);
	ASSERT_EQ(cellSizeArr.size(), 3u);
	EXPECT_EQ(cellSizeArr[0], 2);
	EXPECT_EQ(cellSizeArr[1], 1);
	EXPECT_EQ(cellSizeArr[2], 3);
}

// ─── makeGatewayIndex ────────────────────────────────────────────────────────

// Empty input → empty output.
TEST(MakeGatewayIndex, EmptyInputProducesEmptyOutput) {
	auto cellSizeArr  = MakeIntVec({});
	auto gatewayIndex = MakeIntVec({});
	clast::hash::makeGatewayIndex(cellSizeArr, gatewayIndex);
	ASSERT_EQ(gatewayIndex.size(), 0u);
}

// Single bucket: exclusive scan of {n} → {0}.
TEST(MakeGatewayIndex, SingleBucketStartsAtZero) {
	auto cellSizeArr  = MakeIntVec({5});
	auto gatewayIndex = MakeIntVec({});
	clast::hash::makeGatewayIndex(cellSizeArr, gatewayIndex);
	ASSERT_EQ(gatewayIndex.size(), 1u);
	EXPECT_EQ(gatewayIndex[0], 0);
}

// Uniform buckets: exclusive prefix sum of {1,1,1} → {0,1,2}.
TEST(MakeGatewayIndex, UniformBucketsProduceConsecutiveIndices) {
	auto cellSizeArr  = MakeIntVec({1, 1, 1});
	auto gatewayIndex = MakeIntVec({});
	clast::hash::makeGatewayIndex(cellSizeArr, gatewayIndex);
	ASSERT_EQ(gatewayIndex.size(), 3u);
	EXPECT_EQ(gatewayIndex[0], 0);
	EXPECT_EQ(gatewayIndex[1], 1);
	EXPECT_EQ(gatewayIndex[2], 2);
}

// Non-uniform buckets: {2,1,3} → {0,2,3}.
TEST(MakeGatewayIndex, NonUniformBucketsExclusiveScan) {
	auto cellSizeArr  = MakeIntVec({2, 1, 3});
	auto gatewayIndex = MakeIntVec({});
	clast::hash::makeGatewayIndex(cellSizeArr, gatewayIndex);
	ASSERT_EQ(gatewayIndex.size(), 3u);
	EXPECT_EQ(gatewayIndex[0], 0);
	EXPECT_EQ(gatewayIndex[1], 2);
	EXPECT_EQ(gatewayIndex[2], 3);
}

// ─── removeRepeatedSequence ──────────────────────────────────────────────────

// Empty input → no crash.
TEST(RemoveRepeatedSequence, EmptyInputUnchanged) {
	auto v = MakeIntVec({});
	clast::hash::removeRepeatedSequence(5, v);
	ASSERT_EQ(v.size(), 0u);
}

// All below cutRepeat → none zeroed.
TEST(RemoveRepeatedSequence, AllBelowCutoffKeptIntact) {
	auto v = MakeIntVec({1, 2, 3});
	clast::hash::removeRepeatedSequence(10, v);
	EXPECT_EQ(v[0], 1); EXPECT_EQ(v[1], 2); EXPECT_EQ(v[2], 3);
}

// At and above cutRepeat → zeroed; below → kept.
// cutRepeat=3: {1,2,3,4,5} → {1,2,0,0,0}
TEST(RemoveRepeatedSequence, ZerosAtAndAboveCutoff) {
	auto v = MakeIntVec({1, 2, 3, 4, 5});
	clast::hash::removeRepeatedSequence(3, v);
	EXPECT_EQ(v[0], 1);
	EXPECT_EQ(v[1], 2);
	EXPECT_EQ(v[2], 0);
	EXPECT_EQ(v[3], 0);
	EXPECT_EQ(v[4], 0);
}

// cutRepeat=1: all entries (>= 1) are zeroed.
TEST(RemoveRepeatedSequence, AllZeroedWhenCutoffIsOne) {
	auto v = MakeIntVec({1, 2, 3});
	clast::hash::removeRepeatedSequence(1, v);
	EXPECT_EQ(v[0], 0); EXPECT_EQ(v[1], 0); EXPECT_EQ(v[2], 0);
}

// ─── createHashIndex ─────────────────────────────────────────────────────────
//
// 2-bit encoding: A=00, C=01, G=10, T=11.
// Each k-mer is encoded left-to-right: the first character occupies the
// most-significant bit pair.
// K-mers containing non-ACGT characters are encoded as -1.
// strideLength == lMerLength throughout (non-overlapping windows).

static thrust::host_vector<char> MakeCharVec(const char* s) {
	thrust::host_vector<char> v;
	for(; *s; ++s) v.push_back(*s);
	return v;
}

// Empty base array: hashIndex is empty; no crash.
TEST(CreateHashIndex, EmptyBaseArrayProducesEmptyResult) {
	auto base = MakeCharVec("");
	thrust::host_vector<long> result;
	clast::hash::createHashIndex(1, 1, base, result);
	ASSERT_EQ(result.size(), 0u);
}

// lMer=1, stride=1: each character becomes its own hash value.
// A=0, C=1, G=2, T=3.
TEST(CreateHashIndex, SingleCharKmersACGT) {
	auto base = MakeCharVec("ACGT");
	thrust::host_vector<long> result;
	clast::hash::createHashIndex(1, 1, base, result);
	ASSERT_EQ(result.size(), 4u);
	EXPECT_EQ(result[0], 0L); // A
	EXPECT_EQ(result[1], 1L); // C
	EXPECT_EQ(result[2], 2L); // G
	EXPECT_EQ(result[3], 3L); // T
}

// lMer=1, stride=1: non-ACGT character → -1.
TEST(CreateHashIndex, SingleCharNonACGTBecomesMinusOne) {
	auto base = MakeCharVec("AN");
	thrust::host_vector<long> result;
	clast::hash::createHashIndex(1, 1, base, result);
	ASSERT_EQ(result.size(), 2u);
	EXPECT_EQ(result[0],  0L); // A → 0
	EXPECT_EQ(result[1], -1L); // N → -1
}

// lMer=2, stride=2: two-character non-overlapping k-mers.
// "AC" = 00 01 = 1, "GT" = 10 11 = 11.
TEST(CreateHashIndex, TwoCharKmers) {
	auto base = MakeCharVec("ACGT");
	thrust::host_vector<long> result;
	clast::hash::createHashIndex(2, 2, base, result);
	ASSERT_EQ(result.size(), 2u);
	EXPECT_EQ(result[0],  1L); // AC = 0001 = 1
	EXPECT_EQ(result[1], 11L); // GT = 1011 = 11
}

// lMer=3, stride=3: three-character non-overlapping k-mers.
// "ACG" = 00 01 10 = 6, "TAA" = 11 00 00 = 48.
TEST(CreateHashIndex, ThreeCharKmers) {
	auto base = MakeCharVec("ACGTAA");
	thrust::host_vector<long> result;
	clast::hash::createHashIndex(3, 3, base, result);
	ASSERT_EQ(result.size(), 2u);
	EXPECT_EQ(result[0],  6L); // ACG
	EXPECT_EQ(result[1], 48L); // TAA
}

// lMer=3, stride=3: k-mer containing non-ACGT → entire k-mer is -1.
// "ACG"=6, "NAA"=-1.
TEST(CreateHashIndex, KmerWithOddBaseBecomesMinusOne) {
	auto base = MakeCharVec("ACGNAA");
	thrust::host_vector<long> result;
	clast::hash::createHashIndex(3, 3, base, result);
	ASSERT_EQ(result.size(), 2u);
	EXPECT_EQ(result[0],  6L); // ACG = 6
	EXPECT_EQ(result[1], -1L); // NAA → -1
}

// All-T k-mer: "TTT" = 11 11 11 = 63.
TEST(CreateHashIndex, AllTKmerEncoding) {
	auto base = MakeCharVec("TTT");
	thrust::host_vector<long> result;
	clast::hash::createHashIndex(3, 3, base, result);
	ASSERT_EQ(result.size(), 1u);
	EXPECT_EQ(result[0], 63L);
}

// Trailing characters that don't fill a complete k-mer are ignored
// (integer division in hashIndexSize = size / stride).
TEST(CreateHashIndex, IncompleteTrailingKmerIgnored) {
	// "ACGTT" with lMer=3, stride=3 → only "ACG" fits; "TT" is ignored
	auto base = MakeCharVec("ACGTT");
	thrust::host_vector<long> result;
	clast::hash::createHashIndex(3, 3, base, result);
	ASSERT_EQ(result.size(), 1u);
	EXPECT_EQ(result[0], 6L); // ACG
}
