#pragma once

#include <cuda.h>
#include <vector>

struct Flash_fwd_params
{
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ output_ptr;

    index_t batch;
    index_t seq_len;
    index_t head;
    index_t dim;
    index_t block_M;
    index_t block_N;
    index_t threads;
};

void host_function(Flash_fwd_params params);
void host_function_no_tma(Flash_fwd_params params);

