#include "host/CHostSchedular.cuh"

#include "util/common.hpp"
#include "util/time_attack.hpp"

#include "device/CDeviceHashTable.cuh"
#include "device/hit/CDeviceHitList.cuh"
#include "device/seq/query.cuh"

#include <iostream>
#include <memory>


CHostSchedular::CHostSchedular(
		const CHostSetting& s,
		const CHostSeqList_target& t,
		const CHostSeqList_query& q)
		: setting(s), targetList(t), queryList(q) {}

void CHostSchedular::search(CHostResultHolder& holder) {
	time_attack::print_searching_banner();

	/* scheduling(target) */
	[[maybe_unused]] int searchTimes_target = 0;
	const int tEndID = targetList.getGatewaySize() - 1;
	for(int t_begin = 0, t_end = 1; t_begin < tEndID; t_begin = t_end++) {
		clast::schedular::moveTargetWindow(
				targetList,
				setting.getTargetVRAMSize(),
				tEndID,
				t_begin,
				t_end);
		std::unique_ptr<CDeviceHashTable> pHash;
		time_attack::runLabeledPrefix(
				[&, t_begin, t_end](std::ostream& os) {
					os << " Creating hash table [" << searchTimes_target++ << "]"
					   << " (t_begin/t_end -> " << t_begin << "/" << t_end << ") ";
				},
				"...finished.\n",
				[&] { pHash = std::make_unique<CDeviceHashTable>(setting, targetList, t_begin, t_end); });
		CDeviceHashTable& hashTable = *pHash;

		/* scheduling(query) */
		[[maybe_unused]] int searchTimes_query = 0;
		// this magic number 2 = number of query strand types ('+', '-')
		const int qEndID = (queryList.getGatewaySize()-1)/2;
		for(int q_begin = 0, q_end = 1; q_begin < qEndID; q_begin = q_end++) {
			clast::schedular::moveQueryWindow(
					queryList,
					setting.getQueryVRAMSize(),
					qEndID,
					q_begin,
					q_end);
			std::unique_ptr<CDeviceSeqList_query> pQuery;
			time_attack::runLabeledPrefix(
					[](std::ostream& os) { os << std::endl << "  ...Transferring queries from host to device"; },
					"..............finished.",
					[&] {
						pQuery = std::make_unique<CDeviceSeqList_query>(setting, &queryList, q_begin, q_end);
					});
			CDeviceSeqList_query& deviceQueryList = *pQuery;

			std::unique_ptr<CDeviceHitList> pHit;
			time_attack::runLabeledPrefix(
					[&, q_begin, q_end](std::ostream& os) {
						os << " Creating hit list   [" << searchTimes_query++ << "]"
						   << " (q_begin/q_end -> " << q_begin << "/" << q_end << ") ";
					},
					"...finished.\n",
					[&] {
						// this magic number 2 = number of query strand types ('+', '-')
						pHit = std::make_unique<CDeviceHitList>(setting, hashTable, deviceQueryList, t_begin, q_begin*2);
					});
			CDeviceHitList& hitList = *pHit;

			hitList.getResult(holder); // this subroutine calls holder::addResult().
			holder.addLabel   (targetList);
			holder.addStartIdx(targetList);
		}
	}
}
