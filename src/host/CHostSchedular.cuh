#ifndef C_HOST_SCHEDULAR_CUH_
#define C_HOST_SCHEDULAR_CUH_

#include "host/CHostResultHolder.cuh"
#include "host/CHostSetting.cuh"
#include "host/seq/query.cuh"
#include "host/seq/target.cuh"

class CHostSchedular {
	const CHostSetting&    setting;
	const CHostSeqList_target& targetList;
	const CHostSeqList_query&  queryList;
public:
	CHostSchedular(
			const CHostSetting& s,
			const CHostSeqList_target& t,
			const CHostSeqList_query& q);
	void search(CHostResultHolder& holder);
};


namespace clast::schedular {

template <class SeqList>
void moveQueryWindow(
		const SeqList& queryList,
		const int queryVRAMSize,
		const int qEndID,
		const int q_begin,
		int& q_end) {
	// these magic number 2 = number of query strand types ('+', '-')
	const int q_beginIdx = queryList.getGatewayIdx(q_begin*2);
	while((q_end < qEndID) && (queryList.getGatewayIdx(q_end*2+2) - q_beginIdx < queryVRAMSize)) {
		++q_end;
	}
}

template <class SeqList>
void moveTargetWindow(
		const SeqList& targetList,
		const int targetVRAMSize,
		const int tEndID,
		const int t_begin,
		int& t_end) {
	const int t_beginIdx = targetList.getGatewayIdx(t_begin);
	while((t_end < tEndID) && (targetList.getGatewayIdx(t_end+1) - t_beginIdx < targetVRAMSize)) {
		++t_end;
	}
}

} // namespace clast::schedular

#endif
