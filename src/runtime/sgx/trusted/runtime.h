/*!
 *  Copyright (c) 2018 by Contributors
 * \file trusted/runtime.h
 * \brief TVM SGX trusted API.
 */
#ifndef TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
#define TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_

#include <tvm/runtime/registry.h>
#include <sgx_edger8r.h>
#include "../common.h"

/*!
 * \brief Register a function globally.
 * \code
 *   TVM_REGISTER_SGX_FUNC("DoThing")
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *   });
 * \endcode
 */
#define TVM_REGISTER_ENCLAVE_FUNC(OpName)                              \
  static sgx_status_t _sgx_status = tvm_ocall_register_func(OpName);   \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) =                  \
      ::tvm::runtime::Registry::Register(                              \
          tvm::runtime::sgx::ECALL_PACKED_PFX + OpName)

#endif  // TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
