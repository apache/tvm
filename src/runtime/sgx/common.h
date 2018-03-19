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

static const std::string ECALL_PACKED_PFX = "__ECall_";  // NOLINT(*)

}
}
}

#endif  // TVM_RUNTIME_SGX_COMMON_H_
