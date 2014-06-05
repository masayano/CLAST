#include "CHostMapper.cuh"

#include "CHostSchedular.cuh"

#include "common.hpp"

#include <iostream>

CHostMapper::CHostMapper(const CHostSetting& s) : setting(s) {}

void CHostMapper::addTarget(const std::vector<CHostFASTA>& t) {
	#ifdef TIME_ATTACK
		float elapsed_time_ms=0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate( &start );
		cudaEventCreate( &stop  );
		cudaEventRecord( start, 0 );
		std::cout << "...Preparing target on host";
	#endif /* TIME_ATTACK */

	targetList.add(setting, t);

	#ifdef TIME_ATTACK
		std::cout << "..finished.";
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		std::cout << " (costs " << elapsed_time_ms << "ms)" << std::endl;
	#endif /* TIME_ATTACK */
}

void CHostMapper::addQuery(const std::vector<CHostFASTA>& q) {
	#ifdef TIME_ATTACK
		float elapsed_time_ms=0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate( &start );
		cudaEventCreate( &stop  );
		cudaEventRecord( start, 0 );
		std::cout << "...Preparing query on host";
	#endif /* TIME_ATTACK */

	queryList.add(setting, q);

	#ifdef TIME_ATTACK
		std::cout << "...finished.";
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		std::cout << " (costs " << elapsed_time_ms << "ms)" << std::endl;
	#endif /* TIME_ATTACK */
}

void CHostMapper::getResult(CHostResultHolder& holder) const {
	#ifdef TIME_ATTACK
		float elapsed_time_ms=0.0f;
		cudaEvent_t start, stop;
		cudaEventCreate( &start );
		cudaEventCreate( &stop  );
		cudaEventRecord( start, 0 );
	#endif /* TIME_ATTACK */

	CHostSchedular schedular(setting, targetList, queryList);
	schedular.search(holder);

	#ifdef TIME_ATTACK
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed_time_ms, start, stop );
		std::cout << "...costs " << elapsed_time_ms/1000 << "seconds" << std::endl;
	#endif /* TIME_ATTACK */
}
