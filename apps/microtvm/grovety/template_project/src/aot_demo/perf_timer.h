#ifndef TVM_APPS_MICROTVM_GROVETY_PERF_TIMER_H_
#define TVM_APPS_MICROTVM_GROVETY_PERF_TIMER_H_


#include <stdint.h>


enum PERF_TIMER_OPS {
#ifdef GROVETY_OP_BENCHMARK
    PERF_TIMER_OP_GEMM = 0,
    PERF_TIMER_OP_MAX_POOL,
    PERF_TIMER_OP_AVG_POOL,
    PERF_TIMER_OP_RELU,
#endif // GROVETY_OP_BENCHMARK

    // Must be the latest string
    PERF_TIMER_TOTAL
};

#define PERF_TIMER_NUMBER_OPS (PERF_TIMER_TOTAL + 1)

void perf_timer_start(uint32_t op_id);
void perf_timer_stop(uint32_t op_id);

void perf_timer_clear_all();
uint64_t perf_timer_get_counter(uint32_t op_id);


#endif // TVM_APPS_MICROTVM_GROVETY_PERF_TIMER_H_
