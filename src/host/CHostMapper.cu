#include "host/CHostMapper.cuh"

#include "host/CHostSchedular.cuh"

#include "util/common.hpp"
#include "util/time_attack.hpp"

#include <iostream>

CHostMapper::CHostMapper(const CHostSetting& s) : setting(s) {}

void CHostMapper::addTarget(const std::vector<CHostFASTA>& t) {
	time_attack::runLabeled(
			"...Preparing target on host", "..finished.",
			[&] { targetList.add(setting, t); });
}

void CHostMapper::addQuery(const std::vector<CHostFASTA>& q) {
	time_attack::runLabeled(
			"...Preparing query on host", "...finished.",
			[&] { queryList.add(setting, q); });
}

void CHostMapper::getResult(CHostResultHolder& holder) const {
	time_attack::runEndSeconds([&] {
	CHostSchedular schedular(setting, targetList, queryList);
	schedular.search(holder);
	});
}
