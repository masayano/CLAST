#ifndef TIME_ATTACK_HPP_
#define TIME_ATTACK_HPP_

#include "util/common.hpp"

#include <string_view>
#include <utility>
#ifdef TIME_ATTACK
#include <cuda_runtime.h>
#include <iostream>
#include <ostream>
#endif

namespace time_attack {

/* Whether to append std::endl after the "(costs X ms)" line (when there is no extra tail). */
enum class NewlineAfterCosts { Yes, No };

#ifdef TIME_ATTACK
inline void print_searching_banner() {
  std::cout << std::endl << "...Searching." << std::endl;
}
#else
inline void print_searching_banner() {
}
#endif

#ifdef TIME_ATTACK

namespace detail {

struct CudaEventPair {
  cudaEvent_t start_{}, stop_{};

  CudaEventPair() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  CudaEventPair(const CudaEventPair&) = delete;
  CudaEventPair& operator=(const CudaEventPair&) = delete;

  void recordStart() { cudaEventRecord(start_, 0); }
  float recordStopAndElapsedMs() {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }
  ~CudaEventPair() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
};

}  // namespace detail

/**
 * Manual GPU timer for call sites that cannot use runLabeled (e.g. a local must be declared
 * between "start" output and the end of the interval).
 */
using CudaEventTimer = detail::CudaEventPair;

/** Start message as string, then f(), then between, then (costs X ms). */
template <typename F>
void runLabeled(std::string_view start,
                std::string_view between,
                F&& f,
                NewlineAfterCosts nl = NewlineAfterCosts::Yes) {
  detail::CudaEventPair ev;
  ev.recordStart();
  std::cout << start;
  f();
  std::cout << between;
  const float ms = ev.recordStopAndElapsedMs();
  std::cout << " (costs " << ms << "ms)";
  if (nl == NewlineAfterCosts::Yes) {
    std::cout << std::endl;
  }
}

/** Dynamic start line via void(ostream&), then f(), then string between, then (costs). */
template <typename P, typename F>
void runLabeledPrefix(P&& printStart, std::string_view between, F&& f,
                      NewlineAfterCosts nl = NewlineAfterCosts::Yes) {
  detail::CudaEventPair ev;
  ev.recordStart();
  printStart(std::cout);
  f();
  std::cout << between;
  const float ms = ev.recordStopAndElapsedMs();
  std::cout << " (costs " << ms << "ms)";
  if (nl == NewlineAfterCosts::Yes) {
    std::cout << std::endl;
  }
}

/**
 * (costs X ms) then afterCosts(ostream, ms) for suffix like "N hits found." + endl.
 */
template <typename F, typename T>
void runLabeledWithSuffix(std::string_view start,
                          std::string_view between,
                          F&& f,
                          T&& afterCosts) {
  detail::CudaEventPair ev;
  ev.recordStart();
  std::cout << start;
  f();
  std::cout << between;
  const float ms = ev.recordStopAndElapsedMs();
  std::cout << " (costs " << ms << "ms) ";
  afterCosts(std::cout, ms);
}

/** No start line: record, f(), then ...costs X seconds. */
template <typename F>
void runEndSeconds(F&& f) {
  detail::CudaEventPair ev;
  ev.recordStart();
  f();
  const float ms = ev.recordStopAndElapsedMs();
  std::cout << "...costs " << (ms / 1000.0) << "seconds" << std::endl;
}

#else  // !TIME_ATTACK

struct CudaEventTimer {
  void recordStart() {}
  float recordStopAndElapsedMs() { return 0.f; }
};

template <typename F>
void runLabeled(std::string_view, std::string_view, F&& f, NewlineAfterCosts = NewlineAfterCosts::Yes) {
  std::forward<F>(f)();
}

template <typename P, typename F>
void runLabeledPrefix(P&&, std::string_view, F&& f, NewlineAfterCosts = NewlineAfterCosts::Yes) {
  std::forward<F>(f)();
}

template <typename F, typename T>
void runLabeledWithSuffix(std::string_view, std::string_view, F&& f, T&&) {
  std::forward<F>(f)();
}

template <typename F>
void runEndSeconds(F&& f) {
  std::forward<F>(f)();
}

#endif  // TIME_ATTACK

}  // namespace time_attack

#endif  // TIME_ATTACK_HPP_
