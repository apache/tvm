/*!
 *  Copyright (c) 2017 by Contributors
 * \file module_util.h
 * \brief Helper utilities for module building
 */
#ifndef TVM_RUNTIME_MODULE_UTIL_H_
#define TVM_RUNTIME_MODULE_UTIL_H_

#include <tvm/runtime/module.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <vector>

extern "C" {
// Function signature for generated packed function in shared library
typedef int (*BackendPackedCFunc)(void* args,
                                  int* type_codes,
                                  int num_args);
}  // extern "C"

namespace tvm {
namespace runtime {
/*!
 * \brief Wrap a BackendPackedCFunc to packed function.
 * \param faddr The function address
 * \param mptr The module pointer node.
 */
PackedFunc WrapPackedFunc(BackendPackedCFunc faddr, const std::shared_ptr<ModuleNode>& mptr);
/*!
 * \brief Load and append module blob to module list
 * \param mblob The module blob.
 * \param module_list The module list to append to
 */
void ImportModuleBlob(const char* mblob, std::vector<Module>* module_list);

/*!
 * \brief Utility to initialize conext function symbols during startup
 * \param flookup A symbol lookup function.
 * \tparam FLookup a function of signature string->void*
 */
template<typename FLookup>
void InitContextFunctions(FLookup flookup) {
  if (auto *fp = reinterpret_cast<decltype(&TVMFuncCall)*>
      (flookup("__TVMFuncCall"))) {
    *fp = TVMFuncCall;
  }
  if (auto *fp = reinterpret_cast<decltype(&TVMAPISetLastError)*>
      (flookup("__TVMAPISetLastError"))) {
    *fp = TVMAPISetLastError;
  }
  if (auto *fp = reinterpret_cast<decltype(&TVMBackendGetFuncFromEnv)*>
      (flookup("__TVMBackendGetFuncFromEnv"))) {
    *fp = TVMBackendGetFuncFromEnv;
  }
  if (auto *fp = reinterpret_cast<decltype(&TVMBackendAllocWorkspace)*>
      (flookup("__TVMBackendAllocWorkspace"))) {
    *fp = TVMBackendAllocWorkspace;
  }
  if (auto *fp = reinterpret_cast<decltype(&TVMBackendFreeWorkspace)*>
      (flookup("__TVMBackendFreeWorkspace"))) {
    *fp = TVMBackendFreeWorkspace;
  }
  if (auto *fp = reinterpret_cast<decltype(&TVMBackendParallelFor)*>
      (flookup("__TVMBackendParallelFor"))) {
    *fp = TVMBackendParallelFor;
  }
}
}  // namespace runtime
}  // namespace tvm
#endif   // TVM_RUNTIME_MODULE_UTIL_H_
