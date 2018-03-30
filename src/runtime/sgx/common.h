/*!
 *  Copyright (c) 2018 by Contributors
 * \file common.h
 * \brief TVM SGX common API.
 */
#ifndef TVM_RUNTIME_SGX_COMMON_H_
#define TVM_RUNTIME_SGX_COMMON_H_

#include <tvm/runtime/registry.h>
#include <string>

namespace tvm {
namespace runtime {
namespace sgx {

#define TVM_SGX_CHECKED_CALL(Function)                                         \
  sgx_status_t TVM_STR_CONCAT(__sgx_status_, __LINE__) = SGX_ERROR_UNEXPECTED; \
  TVM_STR_CONCAT(__sgx_status_, __LINE__) = Function;                          \
  CHECK_EQ(TVM_STR_CONCAT(__sgx_status_, __LINE__), SGX_SUCCESS)               \
    << "SGX Error: " << TVM_STR_CONCAT(__sgx_status_, __LINE__);

static const std::string ECALL_PACKED_PFX = "__ECall_";  // NOLINT(*)

}
}
}

#endif  // TVM_RUNTIME_SGX_COMMON_H_
