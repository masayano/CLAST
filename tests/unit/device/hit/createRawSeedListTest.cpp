// Unit tests for the clast::hit functors and template helpers declared in
// src/device/hit/createRawSeedList.cuh:
//   modifyGatewayKey, fillGatewayIndex, fillCellSize,
//   writeGatewayIndexValues, writeCellSizeValues.
//
// writeGatewayKey (CUDA kernel dispatch) and buildRawSeedList (complex
// integration) are intentionally excluded from unit testing.

#include "device/hit/createRawSeedList.cuh"
#include "hitTestUtil.hpp"

#include <gtest/gtest.h>

using clast::test::hit::MakeIntVec;

// ─── modifyGatewayKey ────────────────────────────────────────────────────────

// flg=true → gatewayKey is passed through unchanged.
TEST(ModifyGatewayKey, ReturnKeyWhenFlagTrue) {
	clast::hit::modifyGatewayKey f;
	EXPECT_EQ(f(5, true),  5);
	EXPECT_EQ(f(0, true),  0);
	EXPECT_EQ(f(42, true), 42);
}

// flg=false → always returns -1 regardless of key value.
TEST(ModifyGatewayKey, ReturnMinusOneWhenFlagFalse) {
	clast::hit::modifyGatewayKey f;
	EXPECT_EQ(f(5,  false), -1);
	EXPECT_EQ(f(0,  false), -1);
	EXPECT_EQ(f(99, false), -1);
}

// ─── fillGatewayIndex ────────────────────────────────────────────────────────

// key == -1 → 0 (no hash hit; first argument is ignored).
TEST(FillGatewayIndex, ReturnsZeroWhenKeyIsMinusOne) {
	clast::hit::fillGatewayIndex f;
	EXPECT_EQ(f(99, -1), 0);
	EXPECT_EQ(f(0,  -1), 0);
}

// key >= 0 → returns the looked-up gatewayIndex value.
TEST(FillGatewayIndex, ReturnsGatewayIndexWhenKeyValid) {
	clast::hit::fillGatewayIndex f;
	EXPECT_EQ(f(7,  0), 7);
	EXPECT_EQ(f(42, 3), 42);
}

// ─── fillCellSize ────────────────────────────────────────────────────────────

// key == -1 → 0 (no hash hit; first argument is ignored).
TEST(FillCellSize, ReturnsZeroWhenKeyIsMinusOne) {
	clast::hit::fillCellSize f;
	EXPECT_EQ(f(10, -1), 0);
	EXPECT_EQ(f(0,  -1), 0);
}

// key >= 0 → returns the looked-up cellSize value.
TEST(FillCellSize, ReturnsCellSizeWhenKeyValid) {
	clast::hit::fillCellSize f;
	EXPECT_EQ(f(3,  0), 3);
	EXPECT_EQ(f(15, 2), 15);
}

// ─── writeGatewayIndexValues ─────────────────────────────────────────────────

// Empty input → output stays empty; no crash.
TEST(WriteGatewayIndexValues, EmptyInputProducesEmptyOutput) {
	auto hash = MakeIntVec({10, 20, 30});
	auto keys = MakeIntVec({});
	auto out  = MakeIntVec({});
	clast::hit::writeGatewayIndexValues(hash, keys, out);
	ASSERT_EQ(out.size(), 0u);
}

// Single entry: key selects the correct element from hash.
TEST(WriteGatewayIndexValues, SingleEntryLooksUpCorrectValue) {
	auto hash = MakeIntVec({10, 20, 30});
	auto keys = MakeIntVec({1});
	auto out  = MakeIntVec({0});
	clast::hit::writeGatewayIndexValues(hash, keys, out);
	EXPECT_EQ(out[0], 20);
}

// Multiple entries with distinct keys: each maps to the right hash element.
TEST(WriteGatewayIndexValues, MultipleEntriesPermuted) {
	// hash[0]=10, hash[1]=20, hash[2]=30
	// keys = {2, 0, 1} → expected = {30, 10, 20}
	auto hash = MakeIntVec({10, 20, 30});
	auto keys = MakeIntVec({2, 0, 1});
	auto out  = MakeIntVec({0, 0, 0});
	clast::hit::writeGatewayIndexValues(hash, keys, out);
	EXPECT_EQ(out[0], 30);
	EXPECT_EQ(out[1], 10);
	EXPECT_EQ(out[2], 20);
}

// Repeated key: multiple entries pointing at the same hash slot.
TEST(WriteGatewayIndexValues, RepeatedKeyProducesSameValue) {
	auto hash = MakeIntVec({5, 6, 7});
	auto keys = MakeIntVec({1, 1, 1});
	auto out  = MakeIntVec({0, 0, 0});
	clast::hit::writeGatewayIndexValues(hash, keys, out);
	EXPECT_EQ(out[0], 6);
	EXPECT_EQ(out[1], 6);
	EXPECT_EQ(out[2], 6);
}

// ─── writeCellSizeValues ─────────────────────────────────────────────────────

// Empty input → output stays empty; no crash.
TEST(WriteCellSizeValues, EmptyInputProducesEmptyOutput) {
	auto hash = MakeIntVec({3, 5, 7});
	auto keys = MakeIntVec({});
	auto out  = MakeIntVec({});
	clast::hit::writeCellSizeValues(hash, keys, out);
	ASSERT_EQ(out.size(), 0u);
}

// Single entry: key selects the correct cell size from hash.
TEST(WriteCellSizeValues, SingleEntryLooksUpCorrectValue) {
	auto hash = MakeIntVec({3, 5, 7});
	auto keys = MakeIntVec({2});
	auto out  = MakeIntVec({0});
	clast::hit::writeCellSizeValues(hash, keys, out);
	EXPECT_EQ(out[0], 7);
}

// Multiple entries with distinct keys: each maps to the right cell size.
TEST(WriteCellSizeValues, MultipleEntriesPermuted) {
	// hash[0]=3, hash[1]=5, hash[2]=7
	// keys = {1, 2, 0} → expected = {5, 7, 3}
	auto hash = MakeIntVec({3, 5, 7});
	auto keys = MakeIntVec({1, 2, 0});
	auto out  = MakeIntVec({0, 0, 0});
	clast::hit::writeCellSizeValues(hash, keys, out);
	EXPECT_EQ(out[0], 5);
	EXPECT_EQ(out[1], 7);
	EXPECT_EQ(out[2], 3);
}

// Repeated key: all entries pointing at the same hash slot.
TEST(WriteCellSizeValues, RepeatedKeyProducesSameValue) {
	auto hash = MakeIntVec({3, 5, 7});
	auto keys = MakeIntVec({0, 0, 0});
	auto out  = MakeIntVec({0, 0, 0});
	clast::hit::writeCellSizeValues(hash, keys, out);
	EXPECT_EQ(out[0], 3);
	EXPECT_EQ(out[1], 3);
	EXPECT_EQ(out[2], 3);
}
