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
 * \file local_session.cc
 * \brief Local session that directs requests to local API.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <memory>
#include "rpc_local_session.h"

namespace tvm {
namespace runtime {

RPCSession::PackedFuncHandle
LocalSession::GetFunction(const std::string& name) {
  PackedFunc pf = this->GetFunctionInternal(name);
  // return raw handl because the remote need to explicitly manage it.
  if (pf != nullptr) return new PackedFunc(pf);
  return nullptr;
}

void LocalSession::CallFunc(RPCSession::PackedFuncHandle func,
                            const TVMValue* arg_values,
                            const int* arg_type_codes,
                            int num_args,
                            const FEncodeReturn& encode_return) {
  auto* pf = static_cast<PackedFunc*>(func);
  TVMRetValue rv;

  pf->CallPacked(TVMArgs(arg_values, arg_type_codes, num_args), &rv);
  int rv_tcode = rv.type_code();

  // return value encoding.
  TVMValue ret_value_pack[3];
  int ret_tcode_pack[3];
  TVMArgsSetter set_arg(ret_value_pack, ret_tcode_pack);
  // first location always encode type code.
  set_arg(0, rv_tcode);

  if (rv_tcode == kTVMNDArrayHandle) {
    // We follow a special protocol to return NDArray to client side
    // The first pack value is the NDArray handle as DLTensor
    // The second pack value is a customized deleter that deletes the NDArray.
    rv.MoveToCHost(&ret_value_pack[1], &ret_tcode_pack[1]);
    ret_tcode_pack[1] = kTVMDLTensorHandle;
    ret_value_pack[2].v_handle = ret_value_pack[1].v_handle;
    ret_tcode_pack[2] = kTVMOpaqueHandle;
    encode_return(TVMArgs(ret_value_pack, ret_tcode_pack, 3));
  } else if (rv_tcode == kTVMPackedFuncHandle ||
             rv_tcode == kTVMModuleHandle) {
    // MoveToCHost means rv no longer manages the object.
    // return handle instead.
    rv.MoveToCHost(&ret_value_pack[1], &ret_tcode_pack[1]);
    ret_tcode_pack[1] = kTVMOpaqueHandle;
    encode_return(TVMArgs(ret_value_pack, ret_tcode_pack, 2));
  } else if (rv_tcode == kTVMBytes) {
    TVMByteArray byte_arr;
    auto* sptr = rv.ptr<std::string>();
    byte_arr.data = sptr->data();
    byte_arr.size = sptr->length();
    set_arg(1, byte_arr);
    encode_return(TVMArgs(ret_value_pack, ret_tcode_pack, 2));
  } else {
    set_arg(1, rv);
    encode_return(TVMArgs(ret_value_pack, ret_tcode_pack, 2));
  }
}

void LocalSession::CopyToRemote(void* from,
                                size_t from_offset,
                                void* to,
                                size_t to_offset,
                                size_t nbytes,
                                TVMContext ctx_to,
                                DLDataType type_hint) {
  TVMContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  this->GetDeviceAPI(ctx_to)->CopyDataFromTo(
      from, from_offset,
      to, to_offset,
      nbytes, cpu_ctx, ctx_to, type_hint, nullptr);
}

void LocalSession::CopyFromRemote(void* from,
                                  size_t from_offset,
                                  void* to,
                                  size_t to_offset,
                                  size_t nbytes,
                                  TVMContext ctx_from,
                                  DLDataType type_hint) {
  TVMContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;

  this->GetDeviceAPI(ctx_from)->CopyDataFromTo(
      from, from_offset,
      to, to_offset,
      nbytes, ctx_from, cpu_ctx, type_hint, nullptr);
}

void LocalSession::FreeHandle(void* handle, int type_code) {
  TVMValue value;
  value.v_handle = handle;
  // will trigger deleter once the rv goes out of the scope.
  TVMRetValue rv = TVMRetValue::MoveFromCHost(value, type_code);
}

DeviceAPI* LocalSession::GetDeviceAPI(TVMContext ctx, bool allow_missing) {
  return DeviceAPI::Get(ctx, allow_missing);
}

PackedFunc LocalSession::GetFunctionInternal(const std::string& name) {
  auto* fp = tvm::runtime::Registry::Get(name);
  if (fp != nullptr) {
    return *fp;
  } else {
    return nullptr;
  }
}

TVM_REGISTER_GLOBAL("rpc.LocalSession")
.set_body_typed([]() {
  return CreateRPCSessionModule(std::make_shared<LocalSession>());
});

}  // namespace runtime
}  // namespace tvm
