#ifndef TVM_APPS_MICROTVM_GROVETY_PERF_TIMER_H_
#define TVM_APPS_MICROTVM_GROVETY_PERF_TIMER_H_

#ifdef GROVETY_PERF_TIMER

#include <stdint.h>


enum PERF_TIMER_OPS {
    PERF_TIMER_OP_CONV = 0,
    PERF_TIMER_OP_MAX_POOL,


    // Must be the latest string
    PERF_TIMER_TOTAL
};

void perf_timer_start(uint32_t op_id);
void perf_timer_stop(uint32_t op_id);

void perf_timer_clear_all();
uint64_t perf_timer_get_counter(uint32_t op_id);


#endif // GROVETY_PERF_TIMER

#endif // TVM_APPS_MICROTVM_GROVETY_PERF_TIMER_H_
