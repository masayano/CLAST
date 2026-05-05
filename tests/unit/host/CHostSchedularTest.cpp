#include "host/CHostSchedular.cuh"

#include <gtest/gtest.h>
#include <vector>

struct FakeSeqList {
	std::vector<int> gateway;
	int getGatewayIdx(int seqID) const { return gateway[seqID]; }
};

// ─── moveQueryWindow ─────────────────────────────────────────────────────────
//
// gateway is indexed per strand: gateway[seq*2] is the byte offset of the
// '+' strand of sequence seq; gateway[seq*2+2] is the start of seq+1.
// queryVRAMSize is compared against (gateway[q_end*2+2] - gateway[q_begin*2]).

// q_end already equals qEndID on entry → no increment.
TEST(MoveQueryWindow, AlreadyAtEndID_NoAdvance) {
	FakeSeqList q;
	q.gateway = {0, 5, 10};   // 1 sequence of 10 bytes
	int q_end = 1;
	clast::schedular::moveQueryWindow(q, /*vramSize=*/100, /*qEndID=*/1, /*q_begin=*/0, q_end);
	EXPECT_EQ(q_end, 1);
}

// The next sequence would push total over vramSize → stays at initial q_end.
TEST(MoveQueryWindow, NextSequenceExceedsVRAM_NoAdvance) {
	FakeSeqList q;
	// 2 sequences, 10 bytes each (stride 5 per strand)
	q.gateway = {0, 5, 10, 15, 20};
	int q_end = 1;
	clast::schedular::moveQueryWindow(q, /*vramSize=*/5, /*qEndID=*/2, /*q_begin=*/0, q_end);
	EXPECT_EQ(q_end, 1);
}

// vramSize large enough for exactly one more sequence → q_end advances by 1.
TEST(MoveQueryWindow, ExactlyOneSequenceFits) {
	FakeSeqList q;
	// 3 sequences, 10 bytes each
	q.gateway = {0, 5, 10, 15, 20, 25, 30};
	int q_end = 1;
	clast::schedular::moveQueryWindow(q, /*vramSize=*/25, /*qEndID=*/3, /*q_begin=*/0, q_end);
	// gateway[1*2+2]=20 < 25 → advance; gateway[2*2+2]=30 >= 25 → stop
	EXPECT_EQ(q_end, 2);
}

// vramSize large enough for two more sequences → q_end advances by 2.
TEST(MoveQueryWindow, TwoSequencesFit) {
	FakeSeqList q;
	// 3 sequences, 10 bytes each
	q.gateway = {0, 5, 10, 15, 20, 25, 30};
	int q_end = 1;
	clast::schedular::moveQueryWindow(q, /*vramSize=*/35, /*qEndID=*/3, /*q_begin=*/0, q_end);
	// gateway[4]=20<35 → advance; gateway[6]=30<35 → advance; q_end=3==qEndID → stop
	EXPECT_EQ(q_end, 3);
}

// vramSize fits everything but loop stops at qEndID.
TEST(MoveQueryWindow, StopsAtEndID) {
	FakeSeqList q;
	// 3 sequences, 10 bytes each
	q.gateway = {0, 5, 10, 15, 20, 25, 30};
	int q_end = 1;
	clast::schedular::moveQueryWindow(q, /*vramSize=*/1000, /*qEndID=*/3, /*q_begin=*/0, q_end);
	EXPECT_EQ(q_end, 3);
}

// Non-zero q_begin: base offset shifts, window size computed from q_begin.
TEST(MoveQueryWindow, NonZeroQBegin) {
	FakeSeqList q;
	// 3 sequences, 10 bytes each
	q.gateway = {0, 5, 10, 15, 20, 25, 30};
	int q_end = 2;
	// q_beginIdx = gateway[1*2] = 10; gateway[2*2+2]-10 = 30-10 = 20 < 25 → advance
	clast::schedular::moveQueryWindow(q, /*vramSize=*/25, /*qEndID=*/3, /*q_begin=*/1, q_end);
	EXPECT_EQ(q_end, 3);
}

// ─── moveTargetWindow ────────────────────────────────────────────────────────
//
// Targets have one strand each: gateway[t] is the byte offset of target t;
// gateway[t+1] is the start of the next target.
// targetVRAMSize is compared against (gateway[t_end+1] - gateway[t_begin]).

// t_end already equals tEndID on entry → no increment.
TEST(MoveTargetWindow, AlreadyAtEndID_NoAdvance) {
	FakeSeqList t;
	t.gateway = {0, 10};   // 1 sequence of 10 bytes
	int t_end = 1;
	clast::schedular::moveTargetWindow(t, /*vramSize=*/100, /*tEndID=*/1, /*t_begin=*/0, t_end);
	EXPECT_EQ(t_end, 1);
}

// The next sequence would push total over vramSize → stays at initial t_end.
TEST(MoveTargetWindow, NextSequenceExceedsVRAM_NoAdvance) {
	FakeSeqList t;
	// 2 sequences, 10 bytes each
	t.gateway = {0, 10, 20};
	int t_end = 1;
	clast::schedular::moveTargetWindow(t, /*vramSize=*/5, /*tEndID=*/2, /*t_begin=*/0, t_end);
	EXPECT_EQ(t_end, 1);
}

// vramSize large enough for exactly one more sequence → t_end advances by 1.
TEST(MoveTargetWindow, ExactlyOneSequenceFits) {
	FakeSeqList t;
	// 3 sequences, 10 bytes each
	t.gateway = {0, 10, 20, 30};
	int t_end = 1;
	clast::schedular::moveTargetWindow(t, /*vramSize=*/25, /*tEndID=*/3, /*t_begin=*/0, t_end);
	// gateway[2]=20<25 → advance; gateway[3]=30>=25 → stop
	EXPECT_EQ(t_end, 2);
}

// vramSize large enough for two more sequences → t_end advances by 2.
TEST(MoveTargetWindow, TwoSequencesFit) {
	FakeSeqList t;
	// 3 sequences, 10 bytes each
	t.gateway = {0, 10, 20, 30};
	int t_end = 1;
	clast::schedular::moveTargetWindow(t, /*vramSize=*/35, /*tEndID=*/3, /*t_begin=*/0, t_end);
	// gateway[2]=20<35 → advance; gateway[3]=30<35 → advance; t_end=3==tEndID → stop
	EXPECT_EQ(t_end, 3);
}

// vramSize fits everything but loop stops at tEndID.
TEST(MoveTargetWindow, StopsAtEndID) {
	FakeSeqList t;
	// 3 sequences, 10 bytes each
	t.gateway = {0, 10, 20, 30};
	int t_end = 1;
	clast::schedular::moveTargetWindow(t, /*vramSize=*/1000, /*tEndID=*/3, /*t_begin=*/0, t_end);
	EXPECT_EQ(t_end, 3);
}

// Non-zero t_begin: base offset shifts, window size computed from t_begin.
TEST(MoveTargetWindow, NonZeroTBegin) {
	FakeSeqList t;
	// 3 sequences, 10 bytes each
	t.gateway = {0, 10, 20, 30};
	int t_end = 2;
	// t_beginIdx = gateway[1] = 10; gateway[3]-10 = 20 < 25 → advance
	clast::schedular::moveTargetWindow(t, /*vramSize=*/25, /*tEndID=*/3, /*t_begin=*/1, t_end);
	EXPECT_EQ(t_end, 3);
}
