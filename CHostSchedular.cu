#include "CHostSchedular.cuh"

#include "common.hpp"

#include "CDeviceHashTable.cuh"
#include "CDeviceHitList.cuh"
#include "CDeviceSeqList_query.cuh"

#include <iostream>
#include <sstream>

void moveQueryWindow(
		const CHostSeqList_query& queryList,
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

void moveTargetWindow(
		const CHostSeqList_target& targetList,
		const int targetVRAMSize,
		const int tEndID,
		const int t_begin,
		int& t_end) {
	const int t_beginIdx = targetList.getGatewayIdx(t_begin);
	while((t_end < tEndID) && (targetList.getGatewayIdx(t_end+1) - t_beginIdx < targetVRAMSize)) {
		++t_end;
	}
}

CHostSchedular::CHostSchedular(
		const CHostSetting& s,
		const CHostSeqList_target& t,
		const CHostSeqList_query& q)
		: setting(s), targetList(t), queryList(q) {}

void CHostSchedular::search(CHostResultHolder& holder) {
	#ifdef TIME_ATTACK
		float elapsed_time_ms=0.0f;
		cudaEvent_t start, stop;
		std::cout << std::endl << "...Searching." << std::endl;
	#endif /* TIME_ATTACK */

	/* scheduling(target) */
	#ifdef TIME_ATTACK
		int searchTimes_target = 0;
	#endif
	const int tEndID = targetList.getGatewaySize() - 1;
	for(int t_begin = 0, t_end = 1; t_begin < tEndID; t_begin = t_end++) {
		moveTargetWindow(
				targetList,
				setting.getTargetVRAMSize(),
				tEndID,
				t_begin,
				t_end);
		#ifdef TIME_ATTACK
			cudaEventCreate( &start );
			cudaEventCreate( &stop  );
			cudaEventRecord( start, 0 );
			std::cout
					<< " Creating hash table [" << searchTimes_target++ << "]"
					<< " (t_begin/t_end -> " << t_begin << "/" << t_end << ") ";
		#endif /* TIME_ATTACK */
		CDeviceHashTable  hashTable(setting, targetList, t_begin, t_end);
		#ifdef TIME_ATTACK
			std::cout << "...finished." << std::endl;
			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsed_time_ms, start, stop );
			std::cout << " (costs " << elapsed_time_ms << "ms)" << std::endl;
		#endif /* TIME_ATTACK */

		/* scheduling(query) */
		#ifdef TIME_ATTACK
			int searchTimes_query = 0;
		#endif
		// this magic number 2 = number of query strand types ('+', '-')
		const int qEndID = (queryList.getGatewaySize()-1)/2;
		for(int q_begin = 0, q_end = 1; q_begin < qEndID; q_begin = q_end++) {
			moveQueryWindow(
					queryList,
					setting.getQueryVRAMSize(),
					qEndID,
					q_begin,
					q_end);
			#ifdef TIME_ATTACK
				cudaEventCreate( &start );
				cudaEventCreate( &stop  );
				cudaEventRecord( start, 0 );
				std::cout << std::endl << "  ...Transferring queries from host to device";
			#endif /* TIME_ATTACK */
			CDeviceSeqList_query deviceQueryList(setting, &queryList, q_begin, q_end);
			#ifdef TIME_ATTACK
				std::cout << "..............finished.";
				cudaEventRecord( stop, 0 );
				cudaEventSynchronize( stop );
				cudaEventElapsedTime( &elapsed_time_ms, start, stop );
				std::cout << " (costs " << elapsed_time_ms << "ms)" << std::endl;
			#endif /* TIME_ATTACK */

			#ifdef TIME_ATTACK
				cudaEventCreate( &start );
				cudaEventCreate( &stop  );
				cudaEventRecord( start, 0 );
				std::cout
						<< " Creating hit list   [" << searchTimes_query++ << "]"
						<< " (q_begin/q_end -> " << q_begin << "/" << q_end << ") ";
			#endif /* TIME_ATTACK */
			// this magic number 2 = number of query strand types ('+', '-')
			CDeviceHitList hitList(setting, hashTable, deviceQueryList, t_begin, q_begin*2);
			#ifdef TIME_ATTACK
				std::cout << "...finished." << std::endl;
				cudaEventRecord( stop, 0 );
				cudaEventSynchronize( stop );
				cudaEventElapsedTime( &elapsed_time_ms, start, stop );
				std::cout << " (costs " << elapsed_time_ms << "ms)" << std::endl;
			#endif /* TIME_ATTACK */

			hitList.getResult(holder); // this subroutine calls holder::addResult().
			holder.addLabel   (targetList);
			holder.addStartIdx(targetList);
		}
	}
}
