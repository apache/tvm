/*!
 *  Copyright (c) 2018 by Contributors
 * \file trusted/runtime.h
 * \brief TVM SGX trusted API.
 */
#ifndef TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
#define TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_

#include <sgx_edger8r.h>
#include <tvm/runtime/packed_func.h>
#include <string>
#include "../common.h"

namespace tvm {
namespace runtime {
namespace sgx {

template<typename... Args>
TVMRetValue OCallPackedFunc(std::string name, Args&& ...args);

}  // namespace sgx
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_SGX_TRUSTED_RUNTIME_H_
