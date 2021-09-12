#ifdef GROVETY_PERF_TIMER

#include "perf_timer.h"
#include <kernel.h>
#include <assert.h>


#define PERF_TIMER_NUMBER_OPS (PERF_TIMER_TOTAL + 1)

static uint64_t counters[PERF_TIMER_NUMBER_OPS];
static uint32_t counters_start_time[PERF_TIMER_NUMBER_OPS];
static uint8_t counter_started_flag[PERF_TIMER_NUMBER_OPS];

void perf_timer_start(uint32_t op_id)
{
    assert(op_id < PERF_TIMER_NUMBER_OPS);
    assert(counter_started_flag[op_id] == 0);

    counter_started_flag[op_id] = 1;
    counters_start_time[op_id] = k_cycle_get_32();
}

void perf_timer_stop(uint32_t op_id)
{
    uint32_t stop_time = k_cycle_get_32();

    assert(op_id < PERF_TIMER_NUMBER_OPS);
    assert(counter_started_flag[op_id] != 0);

    uint32_t start_time = counters_start_time[op_id];
    uint32_t cycles_spent = stop_time - start_time;

    if (stop_time < start_time) {
        cycles_spent = ~((uint32_t)0) - (start_time - stop_time);
    }

    counters[op_id] += k_cyc_to_ns_near32 (cycles_spent);
    counter_started_flag[op_id] = 0;
}

void perf_timer_clear_all()
{
    for (int i = 0; i < PERF_TIMER_NUMBER_OPS; i++) {
        counters[i] = 0;
        counters_start_time[i] = 0;
        counter_started_flag[i] = 0;
    }
}

uint64_t perf_timer_get_counter(uint32_t op_id)
{
    assert(op_id < PERF_TIMER_NUMBER_OPS);

    return counters[op_id];
}

#endif // GROVETY_PERF_TIMER