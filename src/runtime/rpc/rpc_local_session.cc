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
#include "rpc_local_session.h"

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <memory>
#include <vector>

namespace tvm {
namespace runtime {

RPCSession::PackedFuncHandle LocalSession::GetFunction(const std::string& name) {
  if (auto* fp = tvm::runtime::Registry::Get(name)) {
    // return raw handle because the remote need to explicitly manage it.
    tvm::runtime::TVMRetValue ret;
    ret = *fp;
    TVMValue val;
    int type_code;
    ret.MoveToCHost(&val, &type_code);
    return val.v_handle;
  } else {
    return nullptr;
  }
}

void LocalSession::EncodeReturn(TVMRetValue rv, const FEncodeReturn& encode_return) {
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
  } else if (rv_tcode == kTVMPackedFuncHandle || rv_tcode == kTVMModuleHandle ||
             rv_tcode == kTVMObjectHandle) {
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

void LocalSession::CallFunc(RPCSession::PackedFuncHandle func, const TVMValue* arg_values,
                            const int* arg_type_codes, int num_args,
                            const FEncodeReturn& encode_return) {
  PackedFuncObj* pf = static_cast<PackedFuncObj*>(func);
  TVMRetValue rv;

  // unwrap RPCObjectRef in case we are directly using it to call LocalSession
  std::vector<TVMValue> values(arg_values, arg_values + num_args);
  std::vector<int> type_codes(arg_type_codes, arg_type_codes + num_args);
  TVMArgs args(arg_values, arg_type_codes, num_args);

  for (int i = 0; i < num_args; ++i) {
    if (args[i].IsObjectRef<RPCObjectRef>()) {
      RPCObjectRef obj_ref = args[i];
      values[i].v_handle = obj_ref->object_handle();
      continue;
    }
  }

  pf->CallPacked(TVMArgs(values.data(), type_codes.data(), args.size()), &rv);
  this->EncodeReturn(std::move(rv), encode_return);
}

void LocalSession::CopyToRemote(void* from_bytes, DLTensor* to, uint64_t nbytes) {
  ICHECK_EQ(nbytes, GetDataSize(*to));
  DLTensor from;
  from.data = from_bytes;
  from.device = {kDLCPU, 0};
  from.ndim = to->ndim;
  from.shape = to->shape;
  from.dtype = to->dtype;
  from.strides = nullptr;
  from.byte_offset = 0;
  Device dev_to = to->device;
  this->GetDeviceAPI(dev_to)->CopyDataFromTo(&from, to, nullptr);
  // Copy can happen asynchrously
  // synchronize to make sure that copy is completed
  this->GetDeviceAPI(dev_to)->StreamSync(dev_to, nullptr);
}

void LocalSession::CopyFromRemote(DLTensor* from, void* to_bytes, uint64_t nbytes) {
  ICHECK_EQ(nbytes, GetDataSize(*from));
  DLTensor to;
  to.data = to_bytes;
  to.device = {kDLCPU, 0};
  to.ndim = from->ndim;
  to.shape = from->shape;
  to.dtype = from->dtype;
  to.strides = nullptr;
  to.byte_offset = 0;

  Device dev_from = from->device;
  this->GetDeviceAPI(dev_from)->CopyDataFromTo(from, &to, nullptr);
  // Copy can happen asynchrously
  // synchronize to make sure that copy is completed
  this->GetDeviceAPI(dev_from)->StreamSync(dev_from, nullptr);
}

void LocalSession::FreeHandle(void* handle, int type_code) {
  TVMValue value;
  value.v_handle = handle;
  // will trigger deleter once the rv goes out of the scope.
  TVMRetValue rv = TVMRetValue::MoveFromCHost(value, type_code);
}

DeviceAPI* LocalSession::GetDeviceAPI(Device dev, bool allow_missing) {
  return DeviceAPI::Get(dev, allow_missing);
}

TVM_REGISTER_GLOBAL("rpc.LocalSession").set_body_typed([]() {
  return CreateRPCSessionModule(std::make_shared<LocalSession>());
});

}  // namespace runtime
}  // namespace tvm
