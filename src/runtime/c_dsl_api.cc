/*!
 *  Copyright (c) 2017 by Contributors
 * \file cpu_dsl_api.cc
 * \brief DSL API dispatcher
 */
#include <tvm/runtime/registry.h>
#include <tvm/c_dsl_api.h>
#include "./dsl_api.h"
#include "./runtime_base.h"

namespace tvm {
namespace runtime {

DSLAPI* FindDSLAPI() {
  auto* f = Registry::Get("dsl_api.singleton");
  if (f == nullptr) {
    throw dmlc::Error("TVM runtime only environment,"\
                      " DSL API is not available");
  }
  void* ptr = (*f)();
  return static_cast<DSLAPI*>(ptr);
}

static DSLAPI* GetDSLAPI() {
  static DSLAPI* inst = FindDSLAPI();
  return inst;
}
}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

int TVMNodeFree(NodeHandle handle) {
  API_BEGIN();
  GetDSLAPI()->NodeFree(handle);
  API_END();
}

int TVMNodeTypeKey2Index(const char* type_key,
                         int* out_index) {
  API_BEGIN();
  GetDSLAPI()->NodeTypeKey2Index(type_key, out_index);
  API_END();
}


int TVMNodeGetTypeIndex(NodeHandle handle,
                        int* out_index) {
  API_BEGIN();
  GetDSLAPI()->NodeGetTypeIndex(handle, out_index);
  API_END();
}

int TVMNodeGetAttr(NodeHandle handle,
                   const char* key,
                   TVMValue* out_value,
                   int* out_type_code,
                   int* out_success) {
  API_BEGIN();
  GetDSLAPI()->NodeGetAttr(
      handle, key, out_value, out_type_code, out_success);
  API_END();
}

int TVMNodeListAttrNames(NodeHandle handle,
                         int *out_size,
                         const char*** out_array) {
  API_BEGIN();
  GetDSLAPI()->NodeListAttrNames(
      handle, out_size, out_array);
  API_END();
}
