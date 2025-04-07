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
    Any ret = *fp;
    TVMFFIAny ret_any = ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(ret));
    return ret_any.v_obj;
  } else {
    return nullptr;
  }
}

void LocalSession::EncodeReturn(TVMRetValue rv, const FEncodeReturn& encode_return) {
  AnyView packed_args[3];
  // NOTE: this is the place that we need to handle special RPC-related
  // ABI convention for return value passing that is built on top of Any FFI.
  // We need to encode object pointers as opaque raw pointers for passing
  // TODO(tqchen): move to RPC to new ABI
  if (rv == nullptr) {
    packed_args[0] = static_cast<int32_t>(kTVMNullptr);
    packed_args[1] = rv;
    encode_return(ffi::PackedArgs(packed_args, 2));
  } else if (rv.as<NDArray>()) {
    // We follow a special protocol to return NDArray to client side
    // The first pack value is the NDArray handle as DLTensor
    // The second pack value is a customized deleter that deletes the NDArray.
    TVMFFIAny ret_any = ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(rv));
    void* opaque_handle = ret_any.v_obj;
    packed_args[0] = static_cast<int32_t>(kTVMNDArrayHandle);
    packed_args[1] =
        static_cast<DLTensor*>(ObjectHandleToTVMArrayHandle(static_cast<Object*>(opaque_handle)));
    packed_args[2] = opaque_handle;
    encode_return(ffi::PackedArgs(packed_args, 3));
  } else if (const auto* bytes = rv.as<ffi::BytesObj>()) {
    // always pass bytes as byte array
    packed_args[0] = static_cast<int32_t>(kTVMBytes);
    TVMFFIByteArray byte_arr;
    byte_arr.data = bytes->bytes.data;
    byte_arr.size = bytes->bytes.size;
    packed_args[1] = &byte_arr;
    encode_return(ffi::PackedArgs(packed_args, 2));
  } else if (const auto* str = rv.as<ffi::StringObj>()) {
    // always pass bytes as raw string
    packed_args[0] = static_cast<int32_t>(kTVMStr);
    packed_args[1] = str->bytes.data;
    encode_return(ffi::PackedArgs(packed_args, 2));
  } else if (rv.as<ffi::ObjectRef>()) {
    TVMFFIAny ret_any = ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(rv));
    void* opaque_handle = ret_any.v_obj;
    packed_args[1] = opaque_handle;
    if (ret_any.type_index == ffi::TypeIndex::kTVMFFIRuntimeModule) {
      packed_args[0] = static_cast<int32_t>(kTVMModuleHandle);
    } else if (ret_any.type_index == ffi::TypeIndex::kTVMFFIFunc) {
      packed_args[0] = static_cast<int32_t>(kTVMPackedFuncHandle);
    } else {
      packed_args[0] = static_cast<int32_t>(kTVMObjectHandle);
    }
    encode_return(ffi::PackedArgs(packed_args, 2));
  } else {
    AnyView temp = rv;
    TVMValue val;
    int type_code;
    AnyViewToLegacyTVMArgValue(temp.CopyToTVMFFIAny(), &val, &type_code);
    // normal POD encoding through rv
    packed_args[0] = type_code;
    packed_args[1] = rv;
    encode_return(ffi::PackedArgs(packed_args, 2));
  }
}

void LocalSession::CallFunc(RPCSession::PackedFuncHandle func, ffi::PackedArgs args,
                            const FEncodeReturn& encode_return) {
  ffi::FunctionObj* pf = static_cast<ffi::FunctionObj*>(func);

  Any rv;
  std::vector<AnyView> packed_args(args.size());

  // unwrap RPCObjectRef in case we are directly using it to call LocalSession
  for (int i = 0; i < args.size(); ++i) {
    if (auto opt_rpc_obj = args[i].as<RPCObjectRef>()) {
      packed_args[i] = static_cast<const Object*>(opt_rpc_obj.value()->object_handle());
    } else {
      packed_args[i] = args[i];
    }
  }

  pf->CallPacked(packed_args.data(), packed_args.size(), &rv);
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

void LocalSession::FreeHandle(void* handle) {
  // NOTE: the type code is no longer need during free handle.
  ffi::details::ObjectUnsafe::DecRefObjectHandle(handle);
}

DeviceAPI* LocalSession::GetDeviceAPI(Device dev, bool allow_missing) {
  return DeviceAPI::Get(dev, allow_missing);
}

TVM_REGISTER_GLOBAL("rpc.LocalSession").set_body_typed([]() {
  return CreateRPCSessionModule(std::make_shared<LocalSession>());
});

}  // namespace runtime
}  // namespace tvm
