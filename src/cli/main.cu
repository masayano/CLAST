#include "util/common.hpp"
#include "util/time_attack.hpp"

#include "host/CFASTALoader.hpp"
#include "host/CHostFASTA.hpp"
#include "host/CHostMapper.cuh"
#include "host/CHostSetting.cuh"

#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>

#include "test_support/CTest.cuh"

namespace {

std::vector<CHostFASTA> loadQueryChunk(CFASTALoader& loader, int partIndex) {
	std::vector<CHostFASTA> out;
	time_attack::runLabeledPrefix(
			[partIndex](std::ostream& os) { os << std::endl << "...Loading FASTA data in ./query (part:" << partIndex << ")"; },
			"...finished.",
			[&] { loader.loadFASTA(out); });
	return out;
}

std::vector<CHostFASTA> loadTargetChunk(CFASTALoader& loader, int partIndex) {
	std::vector<CHostFASTA> out;
	time_attack::runLabeledPrefix(
			[partIndex](std::ostream& os) { os << std::endl << "...Loading FASTA data in ./target(part:" << partIndex << ")"; },
			"...finished.",
			[&] { loader.loadFASTA(out); });
	return out;
}

void sleepForGpuCoolDown(int sleepTimeSec) {
	std::cout
			<< " CLAST is now sleeping in order to make GPU cool." << std::endl
			<< " It will cost " << sleepTimeSec << " sec." << std::endl;
	const clock_t s = clock();
	clock_t c;
	do {
		if ((c = clock()) == static_cast<clock_t>(-1)) {
			break;
		}
	} while ((1000UL * (c - s) / CLOCKS_PER_SEC) <= (static_cast<unsigned long>(sleepTimeSec) * 1000UL));
	std::cout << " ...finished." << std::endl << std::endl;
}

}  // namespace

/******************************************* main function ***********************************************/

int main(const int argc, const char** argv) {
	#ifdef MODE_TEST
	CTest::openLogger();
	#endif /* MODE_TEST */

	/* load setting */
	std::cout << "...Loading search setting." << std::endl;
	CHostSetting setting(argc, argv);

	/* clear result file */
	std::ofstream ofs(setting.getOutputFile().c_str());
	ofs.close();

	/* select device */
	cudaSetDevice(setting.getDeviceID());

	/* timer start */
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
	cudaEventRecord( start, 0 );
	std::cout << " << Start searching >>" << std::endl << std::endl;

	CFASTALoader queryFASTALoader(
			setting.getQueryFileArray(),
			setting.getQueryRAMSize() / 2, // magic number "2" ... "+" query and "-" query. 
			"query");
	for(int i = 0; queryFASTALoader.getFileIndex() != -1; ++i) {
		const std::vector<CHostFASTA> queryFASTA = loadQueryChunk(queryFASTALoader, i);
		CHostResultHolder holder(queryFASTA);
		CFASTALoader targetFASTALoader(
				setting.getTargetFileArray(),
				setting.getTargetRAMSize(),
				"target");
		for(int j = 0; targetFASTALoader.getFileIndex() != -1; ++j) {
			const std::vector<CHostFASTA> targetFASTA = loadTargetChunk(targetFASTALoader, j);
			/* mapping */
			CHostMapper mapper(setting);
			mapper.addTarget(targetFASTA);
			mapper.addQuery (queryFASTA);
			mapper.getResult(holder);
		}
		std::cout << " Fixing and printing the results." << std::endl;
		holder.fixResult();
		holder.printResult(setting.getNumberOfOutput(), setting.getOutputFile());
		std::cout << " ...finished." << std::endl;
		sleepForGpuCoolDown(setting.getSleepTime());
	}
	std::cout << std::endl << " << Searching end >>" << std::endl;
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	std::cout << "Total: " << elapsed_time_ms/1000 << "sec." << std::endl;

	return 0;
}
