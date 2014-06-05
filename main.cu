#include "common.hpp"

#include "CFASTALoader.hpp"
#include "CHostFASTA.hpp"
#include "CHostMapper.cuh"
#include "CHostSetting.cuh"

#include <iostream>
#include <fstream>
#include <vector>

#include "CTest.cuh"

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
		#ifdef TIME_ATTACK
			float elapsed_time_ms_2=0.0f;
			cudaEvent_t start_2, stop_2;
			cudaEventCreate( &start_2 );
			cudaEventCreate( &stop_2  );
			cudaEventRecord( start_2, 0 );
			std::cout << std::endl << "...Loading FASTA data in ./query (part:" << i << ")";
		#endif /* TIME_ATTACK */
		std::vector<CHostFASTA> queryFASTA;
		queryFASTALoader.loadFASTA(queryFASTA);
		#ifdef TIME_ATTACK
			std::cout << "...finished.";
			cudaEventRecord( stop_2, 0 );
			cudaEventSynchronize( stop_2 );
			cudaEventElapsedTime( &elapsed_time_ms_2, start_2, stop_2 );
			std::cout << " (costs " << elapsed_time_ms_2 << "ms)" << std::endl;
		#endif /* TIME_ATTACK */
		CHostResultHolder holder(queryFASTA);
		CFASTALoader targetFASTALoader(
				setting.getTargetFileArray(),
				setting.getTargetRAMSize(),
				"target");
		for(int j = 0; targetFASTALoader.getFileIndex() != -1; ++j) {
			#ifdef TIME_ATTACK
				cudaEventCreate( &start_2 );
				cudaEventCreate( &stop_2  );
				cudaEventRecord( start_2, 0 );
				std::cout << std::endl << "...Loading FASTA data in ./target(part:" << j << ")";
			#endif /* TIME_ATTACK */
			/* load target */
			std::vector<CHostFASTA> targetFASTA;
			targetFASTALoader.loadFASTA(targetFASTA);
			#ifdef TIME_ATTACK
				std::cout << "...finished.";
				cudaEventRecord( stop_2, 0 );
				cudaEventSynchronize( stop_2 );
				cudaEventElapsedTime( &elapsed_time_ms_2, start_2, stop_2 );
				std::cout << " (costs " << elapsed_time_ms_2 << "ms)" << std::endl;
			#endif /* TIME_ATTACK */
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
		/* wait to cool down */
		std::cout
				<< " CLAST is now sleeping in order to make GPU cool." << std::endl
				<< " It will cost " << setting.getSleepTime() << " sec." << std::endl;
		clock_t  s = clock();
		clock_t  c;
		do {
			if ((c = clock()) == static_cast<clock_t>(-1)) { break; }
		} while ((1000UL * (c - s) / CLOCKS_PER_SEC) <= (setting.getSleepTime() * 1000UL));
		std::cout << " ...finished." << std::endl << std::endl;
	}
	std::cout << std::endl << " << Searching end >>" << std::endl;
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	std::cout << "Total: " << elapsed_time_ms/1000 << "sec." << std::endl;

	return 0;
}
