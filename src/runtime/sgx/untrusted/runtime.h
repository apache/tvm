/*!
 *  Copyright (c) 2018 by Contributors
 * \file untrusted/runtime.h
 * \brief TVM SGX untrusted API.
 */
#ifndef TVM_RUNTIME_SGX_UNTRUSTED_RUNTIME_H_
#define TVM_RUNTIME_SGX_UNTRUSTED_RUNTIME_H_

#include <sgx_edger8r.h>

extern "C" {
sgx_status_t ecall_tvm_main(sgx_enclave_id_t eid,
                            const void* args,
                            const int* type_codes,
                            int num_args);
sgx_status_t ecall_tvm_run_worker(sgx_enclave_id_t eid, const void* cb);
}

#endif  // TVM_RUNTIME_SGX_UNTRUSTED_RUNTIME_H_
