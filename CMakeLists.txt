cmake_minimum_required(VERSION 3.22)

set(CLAST_CUDA_ARCHITECTURES "75" CACHE STRING
  "Semicolon-separated CUDA architectures for modern Linux builds.")
set(CMAKE_CUDA_ARCHITECTURES "${CLAST_CUDA_ARCHITECTURES}" CACHE STRING
  "CUDA architectures used for CLAST builds." FORCE)

project(CLAST LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED)
include(CTest)

set(CLAST_SOURCES
  CDeviceHashTable.cu
  CDeviceHitList.cu
  CDeviceHitList_alignmentHits.cu
  CDeviceHitList_createRawSeedList.cu
  CDeviceHitList_deleteBadHits.cu
  CDeviceHitList_deleteDuplicateSeeds.cu
  CDeviceHitList_deleteIsolateSeeds.cu
  CDeviceHitList_deleteSeedsOnSequenceBoundary.cu
  CDeviceHitList_sortSeeds.cu
  CDeviceSeqList.cu
  CDeviceSeqList_query.cu
  CDeviceSeqList_target.cu
  CFASTALoader.cpp
  CHostFASTA.cpp
  CHostMapper.cu
  CHostResultHolder.cu
  CHostSchedular.cu
  CHostSeqList.cu
  CHostSeqList_query.cu
  CHostSeqList_target.cu
  CHostSetting.cu
  CTest.cu
  krnlAlignment.cu
  krnlBinarySearch.cu
  krnlCalculateEvalue.cu
  krnlWriteSeedList.cu
  listenCommandLine.cpp
  main.cu
  utilAddSequence.cu
  utilResultSorting.cu
  utilReverseSeq.cpp
)

add_executable(clast ${CLAST_SOURCES})
target_include_directories(clast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(clast PRIVATE CUDA::cudart)
set_target_properties(clast PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(BUILD_TESTING)
  add_test(
    NAME clast_build_artifacts
    COMMAND /usr/bin/env bash
            ${CMAKE_SOURCE_DIR}/tests/smoke/check_build_artifacts.sh
            $<TARGET_FILE:clast>
            ${CMAKE_SOURCE_DIR}/tests/smoke/query.fa
            ${CMAKE_SOURCE_DIR}/tests/smoke/target.fa
  )

  add_test(
    NAME clast_smoke
    COMMAND /usr/bin/env bash
            ${CMAKE_SOURCE_DIR}/tests/smoke/run_smoke_test.sh
            $<TARGET_FILE:clast>
            ${CMAKE_SOURCE_DIR}/tests/smoke/target.fa
            ${CMAKE_SOURCE_DIR}/tests/smoke/query.fa
            ${CMAKE_BINARY_DIR}/clast-smoke.tsv
  )
  set_tests_properties(clast_smoke PROPERTIES SKIP_RETURN_CODE 125)
endif()
