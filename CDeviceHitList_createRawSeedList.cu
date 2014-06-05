#include "CDeviceHitList_createRawSeedList.cuh"

#include "krnlBinarySearch.cuh"
#include "krnlWriteSeedList.cuh"

#include "common.hpp"

#ifdef TIME_ATTACK
	#include <iostream>
#endif /* TIME_ATTACK */

#ifdef MODE_TEST
	#include "CTest.cuh"
#endif /* MODE_TEST */

/********************************** private **********************************/
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

struct modifyGatewayKeyArray {
	__device__ int operator() (const int gatewayKey, const bool flg) const {
		if(flg) {
			return gatewayKey;
		} else {
			return -1;
		}
	}
};

void writeGatewayKey(
		const thrust::device_vector<long>& hashGatewayKeyArray,
		const thrust::device_vector<long>& qHashIndexArray,
		thrust::device_vector<int>& gatewayKeyArray) {
	try {
		using namespace thrust;

		device_vector<int> temp_idx(qHashIndexArray.size());
		sequence(temp_idx.begin(), temp_idx.end());

		device_vector<long> copyQHashIndexArray = qHashIndexArray;
		sort_by_key(copyQHashIndexArray.begin(), copyQHashIndexArray.end(), temp_idx.begin());

		const int blockDim_x = 256;
		const int blockNum = (qHashIndexArray.size() - 1) / blockDim_x + 1;
		const dim3 block(65535, (blockNum - 1) / 65535 + 1, 1);
		binarySearch<<<block, blockDim_x>>>(
			qHashIndexArray.size(),
			hashGatewayKeyArray.size(),
			raw_pointer_cast( &*hashGatewayKeyArray.begin()),
			raw_pointer_cast( &*copyQHashIndexArray.begin()),
			raw_pointer_cast( &*gatewayKeyArray    .begin())
		);

		sort_by_key(temp_idx.begin(), temp_idx.end(), gatewayKeyArray.begin());
	} catch(thrust::system::system_error) {
		throw;
	} catch(std::bad_alloc) {
		throw;
	}
}

struct fillGatewayIndexArray {
	__device__ int operator() (const int gatewayIndex, const int key) const {
		if(key == -1) {
			return 0;
		} else {
			return gatewayIndex;
		}
	}
};

void writeGatewayIndexArray(
		const thrust::device_vector<int>& hashGatewayIndexArray,
		const thrust::device_vector<int>& gatewayKeyArray,
		thrust::device_vector<int>& gatewayIndexArray) {
	try {
		using namespace thrust;

		thrust::transform(
				make_permutation_iterator(
						hashGatewayIndexArray.begin(),
						gatewayKeyArray      .begin()
				),
				make_permutation_iterator(
						hashGatewayIndexArray.begin(),
						gatewayKeyArray      .end()
				),
				gatewayKeyArray.begin(),
				gatewayIndexArray.begin(),
				fillGatewayIndexArray()
		);
	} catch(thrust::system::system_error) {
		throw;
	} catch(std::bad_alloc) {
		throw;
	}
}

struct fillCellSizeArray {
	__device__ int operator() (const int cellSize, const int key) const {
		if(key == -1) {
			return 0;
		} else {
			return cellSize;
		}
	}
};

void writeCellSizeArray(
		const thrust::device_vector<int>& hashCellSizeArray,
		const thrust::device_vector<int>& gatewayKeyArray,
		thrust::device_vector<int>& cellSizeArray) {
	try {
		using namespace thrust;

		thrust::transform(
				make_permutation_iterator(
						hashCellSizeArray.begin(),
						gatewayKeyArray  .begin()
				),
				make_permutation_iterator(
						hashCellSizeArray.begin(),
						gatewayKeyArray  .end()
				),
				gatewayKeyArray.begin(),
				cellSizeArray.begin(),
				fillCellSizeArray()
		);
	} catch(thrust::system::system_error) {
		throw;
	} catch(std::bad_alloc) {
		throw;
	}
}

/********************************** public **********************************/

void createRawSeedList(
		const CHostSetting& s,
		const CDeviceHashTable& h,
		const CDeviceSeqList_query& q,
		const int t_begin,
		const int q_begin,
		thrust::device_vector<int>& seed_targetIDArray,
		thrust::device_vector<int>& seed_targetIndexArray,
		thrust::device_vector<int>& seed_queryIDArray,
		thrust::device_vector<int>& seed_queryIndexArray) {
	try {
		#ifdef TIME_ATTACK
			float elapsed_time_ms=0.0f;
			cudaEvent_t start, stop;
			cudaEventCreate( &start );
			cudaEventCreate( &stop  );
			cudaEventRecord( start, 0 );
			std::cout << std::endl << "  ...creating raw seed list";
		#endif /* TIME_ATTACK */
		using namespace thrust;

		const device_vector<long>& qHashIndexArray = q.getHashIndex();

		device_vector<int> gatewayKeyArray(qHashIndexArray.size());
		writeGatewayKey(
				h.getGatewayKey(),
				qHashIndexArray,
				gatewayKeyArray);

		device_vector<int> gatewayIndexArray(qHashIndexArray.size());
		writeGatewayIndexArray(
				h.getGatewayIndex(),
				gatewayKeyArray,
				gatewayIndexArray);

		device_vector<int> cellSizeArray(qHashIndexArray.size());
		writeCellSizeArray(
				h.getCellSizeArray(),
				gatewayKeyArray,
				cellSizeArray);

		device_vector<int> seedWriteIndexArray(qHashIndexArray.size());
		exclusive_scan(cellSizeArray.begin(), cellSizeArray.end(), seedWriteIndexArray.begin());

		const int seedNum = seedWriteIndexArray.back() + cellSizeArray.back();
		seed_targetIDArray   .resize(seedNum);
		seed_targetIndexArray.resize(seedNum);
		seed_queryIDArray    .resize(seedNum);
		seed_queryIndexArray .resize(seedNum);

		/* wtire seed list */ {
			const int blockDim_x = 256;
			const int blockNum = (qHashIndexArray.size() / blockDim_x) + 1;
			const dim3 block(65535, (blockNum - 1) / 65535 + 1, 1);
			writeSeedList<<<block, blockDim_x>>>(
					qHashIndexArray.size(),
					raw_pointer_cast( &*gatewayIndexArray  .begin() ),
					raw_pointer_cast( &*h.getIndexArray()  .begin() ),
					raw_pointer_cast( &*seedWriteIndexArray.begin() ),
					raw_pointer_cast( &*cellSizeArray      .begin() ),
					raw_pointer_cast( &*h.getTarget().getIDArray()   .begin() ),
					raw_pointer_cast( &*h.getTarget().getIndexArray().begin() ),
					raw_pointer_cast( &*q.getIDArray()   .begin() ),
					raw_pointer_cast( &*q.getIndexArray().begin() ),
					raw_pointer_cast( &*seed_targetIDArray   .begin() ),
					raw_pointer_cast( &*seed_targetIndexArray.begin() ),
					raw_pointer_cast( &*seed_queryIDArray    .begin() ),
					raw_pointer_cast( &*seed_queryIndexArray .begin() )
			);
		}

		#ifdef TIME_ATTACK
			std::cout << "................................finished.";
			cudaEventRecord( stop, 0 );
			cudaEventSynchronize( stop );
			cudaEventElapsedTime( &elapsed_time_ms, start, stop );
			std::cout
					<< " (costs " << elapsed_time_ms << "ms) "
					<< seed_targetIDArray.size() << " hits found."
					<< std::endl;
		#endif /* TIME_ATTACK */
		#ifdef MODE_TEST
			CTest::printQueryToTarget(
					seed_targetIDArray,
					seed_targetIndexArray,
					seed_queryIDArray,
					seed_queryIndexArray);
		#endif /* MODE_TEST */
	} catch(thrust::system::system_error) {
		throw;
	} catch(std::bad_alloc) {
		throw;
	}
}
