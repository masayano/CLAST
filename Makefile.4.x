# Add source files here
EXECUTABLE := clast
# CUDA source files (compiled with cudacc)
CUFILES_sm_20 := \
		main.cu \
		CDeviceHashTable.cu \
		CDeviceHitList.cu \
		CDeviceHitList_alignmentHits.cu \
		CDeviceHitList_createRawSeedList.cu \
		CDeviceHitList_deleteBadHits.cu \
		CDeviceHitList_deleteDuplicateSeeds.cu \
		CDeviceHitList_deleteIsolateSeeds.cu \
		CDeviceHitList_deleteSeedsOnSequenceBoundary.cu \
		CDeviceHitList_sortSeeds.cu \
		CDeviceSeqList.cu \
		CDeviceSeqList_query.cu \
		CDeviceSeqList_target.cu \
		CHostMapper.cu \
		CHostResultHolder.cu \
		CHostSchedular.cu \
		CHostSeqList.cu \
		CHostSeqList_query.cu \
		CHostSeqList_target.cu \
		CHostSetting.cu \
		CTest.cu \
		krnlAlignment.cu \
		krnlBinarySearch.cu \
		krnlCalculateEvalue.cu \
		krnlWriteSeedList.cu \
		utilAddSequence.cu \
		utilResultSorting.cu
# CUDA dependency files
CU_DEPS :=
# C/C++ source files (compiled with gcc / c++)
CCFILES := \
		CFASTALoader.cpp \
		CHostFASTA.cpp \
		listenCommandLine.cpp \
		utilReverseSeq.cpp
# Rules and targets
include ../../common/common.mk
