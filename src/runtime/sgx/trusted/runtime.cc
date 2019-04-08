/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2018 by Contributors
 * \file runtime_t.cc
 */
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include "../../c_runtime_api.cc"
#include "../../cpu_device_api.cc"
#include "../../module.cc"
#include "../../module_util.cc"
#include "../../registry.cc"
#include "../../system_lib_module.cc"
#include "../../thread_pool.cc"
#include "../../workspace_pool.cc"
#include "ecall_registry.h"
#include "runtime.h"
#include "threading_backend.cc"

namespace tvm {
namespace runtime {
namespace sgx {

extern "C" {

void tvm_ecall_init(TVMRetValueHandle ret) {}

void tvm_ecall_packed_func(int func_id,
                           const TVMValue* arg_values,
                           const int* type_codes,
                           int num_args,
                           TVMRetValueHandle ret) {
  const PackedFunc* f = ECallRegistry::Get(func_id);
  CHECK(f != nullptr) << "ecall function not found.";

  TVMRetValue rv;
  f->CallPacked(TVMArgs(arg_values, type_codes, num_args), &rv);

  int ret_type_code = rv.type_code();
  if (ret_type_code == kNull) return;

  TVMValue ret_value;
  if (ret_type_code == kBytes || ret_type_code == kStr) {
    // allocate a buffer in untrusted, copy the values in
    std::string bytes = rv;

    void* ret_buf;
    TVM_SGX_CHECKED_CALL(tvm_ocall_reserve_space(
          &ret_buf, bytes.size() + sizeof(TVMByteArray), sizeof(uint64_t)));

    char* data_buf = static_cast<char*>(ret_buf) + sizeof(TVMByteArray);
    memcpy(data_buf, bytes.data(), bytes.size());

    TVMByteArray* arr = static_cast<TVMByteArray*>(ret_buf);
    arr->data = data_buf;
    arr->size = bytes.size();

    ret_value = TVMValue{.v_handle = arr};
    ret_type_code = kBytes;
  } else {
    rv.MoveToCHost(&ret_value, &ret_type_code);
  }
  TVM_SGX_CHECKED_CALL(tvm_ocall_set_return(ret, &ret_value, &ret_type_code, 1));
}

}  // extern "C"

TVM_REGISTER_ENCLAVE_FUNC("__tvm_main__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  Module mod = (*Registry::Get("module._GetSystemLib"))();
  mod.GetFunction("default_function").CallPacked(args, rv);
});

}  // namespace sgx
}  // namespace runtime
}  // namespace tvm
