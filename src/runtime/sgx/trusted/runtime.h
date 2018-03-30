/*!
 *  Copyright (c) 2018 by Contributors
 * \file trusted/runtime.h
 * \brief TVM SGX trusted API.
 */
#ifndef TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
#define TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_

#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <sgx_edger8r.h>
#include "../common.h"

namespace tvm {
namespace runtime {
namespace sgx {

/*!
 * \brief Register an enclave function globally.
 * \code
 *   TVM_REGISTER_ENCLAVE_FUNC("DoThing")
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *   });
 * \endcode
 */
#define TVM_REGISTER_ENCLAVE_FUNC(OpName)                              \
  static TVM_ATTRIBUTE_UNUSED sgx_status_t                             \
      TVM_STR_CONCAT(_sgx_status_reg_, __COUNTER__) =                  \
          tvm_ocall_register_func(OpName);                             \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) =                  \
      ::tvm::runtime::Registry::Register(                              \
          tvm::runtime::sgx::ECALL_PACKED_PFX + OpName)

extern "C" {

void tvm_ecall_init() {}

void tvm_ecall_packed_func(const char* cname,
                           const TVMValue* arg_values,
                           const int* type_codes,
                           int num_args,
                           void* tvm_ret_val) {
  std::string name = std::string(cname);
  CHECK(name.substr(0, sgx::ECALL_PACKED_PFX.size()) == sgx::ECALL_PACKED_PFX)
    << "Function `" << name << "` is not an enclave export.";
  const PackedFunc* f = Registry::Get(name);
  CHECK(f != nullptr) << "Enclave function not found.";
  f->CallPacked(TVMArgs(arg_values, type_codes, num_args),
      reinterpret_cast<TVMRetValue*>(tvm_ret_val));
}

}  // extern "C"

}  // namespace sgx
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
