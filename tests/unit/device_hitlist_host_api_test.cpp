#include "device/CDeviceHitList_seed_host_api.cuh"
#include "device/CDeviceHitList_sortSeeds.cuh"

#include <gtest/gtest.h>

#include <thrust/host_vector.h>
#include <thrust/tuple.h>

namespace {

thrust::host_vector<int> MakeIntVec(std::initializer_list<int> xs) {
	thrust::host_vector<int> v;
	v.reserve(xs.size());
	for (int x : xs) {
		v.push_back(x);
	}
	return v;
}

thrust::tuple<int, int, int, int> RowAt(
		const thrust::host_vector<int>& tID,
		const thrust::host_vector<int>& tIdx,
		const thrust::host_vector<int>& qID,
		const thrust::host_vector<int>& qIdx,
		int i) {
	return thrust::make_tuple(tID[i], tIdx[i], qID[i], qIdx[i]);
}

void ExpectSortedByMeasureDistance(
		const thrust::host_vector<int>& tID,
		const thrust::host_vector<int>& tIdx,
		const thrust::host_vector<int>& qID,
		const thrust::host_vector<int>& qIdx) {
	ASSERT_EQ(tID.size(), tIdx.size());
	ASSERT_EQ(tID.size(), qID.size());
	ASSERT_EQ(tID.size(), qIdx.size());
	const int n = static_cast<int>(tID.size());
	measure_distance cmp;
	for (int i = 0; i < n - 1; ++i) {
		const auto a = RowAt(tID, tIdx, qID, qIdx, i);
		const auto b = RowAt(tID, tIdx, qID, qIdx, i + 1);
		EXPECT_FALSE(cmp(b, a))
				<< "rows " << i << " and " << (i + 1) << " are out of order";
	}
}

} // namespace

TEST(SortSeedsHost, EmptyUnchanged) {
	auto tID = MakeIntVec({});
	auto tIdx = MakeIntVec({});
	auto qID = MakeIntVec({});
	auto qIdx = MakeIntVec({});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

TEST(SortSeedsHost, SingleRowUnchanged) {
	auto tID = MakeIntVec({1});
	auto tIdx = MakeIntVec({2});
	auto qID = MakeIntVec({3});
	auto qIdx = MakeIntVec({4});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(tID[0], 1);
	EXPECT_EQ(tIdx[0], 2);
	EXPECT_EQ(qID[0], 3);
	EXPECT_EQ(qIdx[0], 4);
}

TEST(SortSeedsHost, OrdersByQueryThenTargetThenDiagonalThenQueryIndex) {
	auto tID = MakeIntVec({0, 0, 1});
	auto tIdx = MakeIntVec({10, 9, 3});
	auto qID = MakeIntVec({1, 0, 0});
	auto qIdx = MakeIntVec({0, 5, 4});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	EXPECT_EQ(qID[0], 0);
	EXPECT_EQ(qID[1], 0);
	EXPECT_EQ(qID[2], 1);
}

TEST(DuplicateSeedsHost, NoOpWhenZeroOrOneRow) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({7});
	auto tIdx = MakeIntVec({11});
	auto qID = MakeIntVec({2});
	auto qIdx = MakeIntVec({5});
	deleteDuplicateSeeds(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	tID.clear();
	tIdx.clear();
	qID.clear();
	qIdx.clear();
	deleteDuplicateSeeds(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

TEST(DuplicateSeedsHost, RemovesNearDuplicateOnSortedInput) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({10, 11});
	auto qID = MakeIntVec({0, 0});
	auto qIdx = MakeIntVec({5, 6});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	deleteDuplicateSeeds(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}

TEST(IsolateSeedsHost, RemovesRowNotNearPreviousInSortedOrder) {
	const int W = 100;
	const int G = 8;
	auto tID = MakeIntVec({0, 0});
	auto tIdx = MakeIntVec({10, 10});
	auto qID = MakeIntVec({0, 1});
	auto qIdx = MakeIntVec({5, 5});
	measureDistanceSorting(tID, tIdx, qID, qIdx);
	ExpectSortedByMeasureDistance(tID, tIdx, qID, qIdx);
	deleteSeedHasNotNearPair(W, G, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
	EXPECT_EQ(qID[0], 0);
}

TEST(BoundaryHost, OnQueryBoundaryRemovesSeedPastSequenceEnd) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({86}); // kMer + idx > length
	onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

TEST(BoundaryHost, OnQueryBoundaryKeepsInteriorSeed) {
	const int kMer = 15;
	const int q_begin = 0;
	auto qLen = MakeIntVec({100});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({84}); // kMer + idx == 99 <= 100
	onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 1u);
}

TEST(BoundaryHost, OnQueryBoundaryRespectsQBeginOffset) {
	const int kMer = 10;
	const int q_begin = 10;
	auto qLen = MakeIntVec({1000});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({0});
	auto qID = MakeIntVec({10});
	auto qIdx = MakeIntVec({991}); // maps to length entry at index 0
	onQueryBoundary(kMer, q_begin, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

TEST(BoundaryHost, OnTargetBoundaryRemovesSeedPastSequenceEnd) {
	const int kMer = 12;
	const int t_begin = 0;
	auto tLen = MakeIntVec({200});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({190});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({0});
	onTargetBoundary(kMer, t_begin, tLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}

TEST(BoundaryHost, OnCornerRemovesWhenHitPatchCrossesCorners) {
	const int gap = 8;
	const int t_begin = 0;
	const int q_begin = 0;
	auto tLen = MakeIntVec({100});
	auto qLen = MakeIntVec({50});
	auto tID = MakeIntVec({0});
	auto tIdx = MakeIntVec({5});
	auto qID = MakeIntVec({0});
	auto qIdx = MakeIntVec({30}); // tHitStartIdx = -25 ; +gap < 0
	onCorner(gap, t_begin, q_begin, tLen, qLen, tID, tIdx, qID, qIdx);
	ASSERT_EQ(tID.size(), 0u);
}
