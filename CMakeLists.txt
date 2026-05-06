cmake_minimum_required(VERSION 3.22)

set(CLAST_CUDA_ARCHITECTURES "75" CACHE STRING
  "Semicolon-separated CUDA architectures for modern Linux builds.")
set(CMAKE_CUDA_ARCHITECTURES "${CLAST_CUDA_ARCHITECTURES}" CACHE STRING
  "CUDA architectures used for CLAST builds." FORCE)

project(CLAST LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED)
include(CTest)

option(CLAST_ENABLE_TIME_ATTACK "Compile clast with TIME_ATTACK (verbose GPU timing output)" OFF)

set(CLAST_SOURCES
  src/device/CDeviceHashTable.cu
  src/device/hit/CDeviceHitList.cu
  src/device/hit/alignmentHits.cu
  src/device/hit/createRawSeedList.cu
  src/device/hit/deleteBadHits.cu
  src/device/hit/deleteDuplicateSeeds.cu
  src/device/hit/deleteIsolateSeeds.cu
  src/device/hit/deleteSeedsOnSequenceBoundary.cu
  src/device/hit/sortSeeds.cu
  src/device/seq/CDeviceSeqList.cu
  src/device/seq/query.cu
  src/device/seq/target.cu
  src/host/CFASTALoader.cpp
  src/host/CHostFASTA.cpp
  src/host/CHostMapper.cu
  src/host/CHostResultHolder.cu
  src/host/CHostSchedular.cu
  src/host/seq/CHostSeqList.cu
  src/host/seq/query.cu
  src/host/seq/target.cu
  src/host/CHostSetting.cu
  src/test_support/CTest.cu
  src/kernel/krnlAlignment.cu
  src/kernel/krnlBinarySearch.cu
  src/kernel/krnlCalculateEvalue.cu
  src/kernel/krnlWriteSeedList.cu
  src/cli/listenCommandLine.cpp
  src/cli/main.cu
  src/util/fastaStringUtil.cpp
  src/util/fastaDataSizeUtil.cpp
  src/util/utilAddSequence.cu
  src/util/utilReverseSeq.cpp
)

add_executable(clast ${CLAST_SOURCES})
target_include_directories(clast PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(CLAST_ENABLE_TIME_ATTACK)
  target_compile_definitions(clast PRIVATE TIME_ATTACK)
endif()
target_link_libraries(clast PRIVATE CUDA::cudart)
set_target_properties(clast PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(clast_unit_tests
    tests/unit/device/hit/sortSeedsTest.cpp
    tests/unit/device/hit/deleteDuplicateSeedsTest.cpp
    tests/unit/device/hit/deleteIsolateSeedsTest.cpp
    tests/unit/device/hit/deleteSeedsOnSequenceBoundaryTest.cpp
    tests/unit/device/hit/deleteBadHitsTest.cpp
    tests/unit/device/hit/createRawSeedListTest.cpp
    tests/unit/device/hit/removeDuplicateResultHitsTest.cpp
    tests/unit/device/hit/alignmentHitsTest.cpp
    tests/unit/device/hit/CDeviceHashTableTest.cpp
    tests/unit/util/resultSortingTest.cpp
    tests/unit/util/utilAddSequenceTest.cpp
    tests/unit/host/CHostSchedularTest.cpp
    tests/unit/host/CHostResultHolderTest.cpp
    src/util/utilAddSequence.cu
  )
  set_source_files_properties(
    tests/unit/device/hit/sortSeedsTest.cpp
    tests/unit/device/hit/deleteDuplicateSeedsTest.cpp
    tests/unit/device/hit/deleteIsolateSeedsTest.cpp
    tests/unit/device/hit/deleteSeedsOnSequenceBoundaryTest.cpp
    tests/unit/device/hit/deleteBadHitsTest.cpp
    tests/unit/device/hit/createRawSeedListTest.cpp
    tests/unit/device/hit/removeDuplicateResultHitsTest.cpp
    tests/unit/device/hit/alignmentHitsTest.cpp
    tests/unit/device/hit/CDeviceHashTableTest.cpp
    tests/unit/util/resultSortingTest.cpp
    tests/unit/util/utilAddSequenceTest.cpp
    tests/unit/host/CHostSchedularTest.cpp
    tests/unit/host/CHostResultHolderTest.cpp
    PROPERTIES LANGUAGE CUDA
  )
  target_include_directories(clast_unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/device/hit
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(clast_unit_tests PRIVATE
    GTest::gtest_main
    CUDA::cudart
  )
  set_target_properties(clast_unit_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  include(GoogleTest)
  gtest_discover_tests(clast_unit_tests)

  # Host-only free functions: no CUDA in these sources; runnable without a GPU.
  add_executable(clast_host_unit_tests
    tests/unit/util/utilReverseSeqTest.cpp
    tests/unit/util/fastaStringUtilTest.cpp
    tests/unit/util/fastaDataSizeTest.cpp
    tests/unit/host/CFASTALoaderTest.cpp
    tests/unit/cli/listenCommandLineTest.cpp
    src/util/utilReverseSeq.cpp
    src/util/fastaStringUtil.cpp
    src/util/fastaDataSizeUtil.cpp
    src/host/CHostFASTA.cpp
    src/host/CFASTALoader.cpp
    src/cli/listenCommandLine.cpp
  )
  target_include_directories(clast_host_unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Boost_INCLUDE_DIRS}
  )
  target_link_libraries(clast_host_unit_tests PRIVATE GTest::gtest_main)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(clast_host_unit_tests PRIVATE stdc++fs)
  endif()
  gtest_discover_tests(clast_host_unit_tests)

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
