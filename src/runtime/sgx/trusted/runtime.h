/*!
 *  Copyright (c) 2018 by Contributors
 * \file trusted/runtime.h
 * \brief TVM SGX trusted API.
 */
#ifndef TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
#define TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_

#include <sgx_edger8r.h>

extern "C" {
sgx_status_t ocall_tvm_thread_group_launch(int num_workers, void* cb);
sgx_status_t ocall_tvm_thread_group_join();
sgx_status_t ocall_tvm_api_set_last_error(const char* err);
}

#endif  // TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
