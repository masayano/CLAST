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
  src/device/CDeviceHashTable.cu
  src/device/CDeviceHitList.cu
  src/device/CDeviceHitList_alignmentHits.cu
  src/device/CDeviceHitList_createRawSeedList.cu
  src/device/CDeviceHitList_deleteBadHits.cu
  src/device/CDeviceHitList_deleteDuplicateSeeds.cu
  src/device/CDeviceHitList_deleteIsolateSeeds.cu
  src/device/CDeviceHitList_deleteSeedsOnSequenceBoundary.cu
  src/device/CDeviceHitList_sortSeeds.cu
  src/device/CDeviceSeqList.cu
  src/device/CDeviceSeqList_query.cu
  src/device/CDeviceSeqList_target.cu
  src/host/CFASTALoader.cpp
  src/host/CHostFASTA.cpp
  src/host/CHostMapper.cu
  src/host/CHostResultHolder.cu
  src/host/CHostSchedular.cu
  src/host/CHostSeqList.cu
  src/host/CHostSeqList_query.cu
  src/host/CHostSeqList_target.cu
  src/host/CHostSetting.cu
  src/test_support/CTest.cu
  src/kernel/krnlAlignment.cu
  src/kernel/krnlBinarySearch.cu
  src/kernel/krnlCalculateEvalue.cu
  src/kernel/krnlWriteSeedList.cu
  src/cli/listenCommandLine.cpp
  src/cli/main.cu
  src/util/utilAddSequence.cu
  src/util/utilResultSorting.cu
  src/util/utilReverseSeq.cpp
)

add_executable(clast ${CLAST_SOURCES})
target_include_directories(clast PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
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
