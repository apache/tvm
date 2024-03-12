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
 * \file rpc_device_api.cc
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <utility>

#include "rpc_session.h"

namespace tvm {
namespace runtime {

class RPCDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {
    auto remote_dev = RemoveRPCSessionMask(dev);
    GetSess(dev)->GetDeviceAPI(remote_dev)->SetDevice(remote_dev);
  }

  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {
    auto remote_dev = RemoveRPCSessionMask(dev);
    GetSess(dev)->GetDeviceAPI(remote_dev)->GetAttr(remote_dev, kind, rv);
  }

  void* AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                       Optional<String> mem_scope) final {
    auto sess = GetSess(dev);
    auto remote_dev = RemoveRPCSessionMask(dev);
    void* data =
        sess->GetDeviceAPI(remote_dev)->AllocDataSpace(remote_dev, ndim, shape, dtype, mem_scope);
    RemoteSpace* space = new RemoteSpace();
    space->data = data;
    space->sess = std::move(sess);
    return space;
  }

  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    auto sess = GetSess(dev);
    auto remote_dev = RemoveRPCSessionMask(dev);
    void* data =
        sess->GetDeviceAPI(remote_dev)->AllocDataSpace(remote_dev, nbytes, alignment, type_hint);

    RemoteSpace* space = new RemoteSpace();
    space->data = data;
    space->sess = std::move(sess);
    return space;
  }
  void FreeDataSpace(Device dev, void* ptr) final {
    RemoteSpace* space = static_cast<RemoteSpace*>(ptr);
    auto remote_dev = RemoveRPCSessionMask(dev);
    try {
      GetSess(dev)->GetDeviceAPI(remote_dev)->FreeDataSpace(remote_dev, space->data);
    } catch (const Error& e) {
      // fault tolerance to remote close.
    }
    delete space;
  }

  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final {
    DLDevice dev_from = from->device;
    DLDevice dev_to = to->device;
    if (IsRPCSessionDevice(dev_from) && IsRPCSessionDevice(dev_to)) {
      ICHECK(dev_from.device_type == dev_to.device_type)
          << "Cannot copy across two different remote session";
      DLTensor from_tensor = *from;
      from_tensor.device = RemoveRPCSessionMask(dev_from);
      from_tensor.data = static_cast<const RemoteSpace*>(from->data)->data;
      DLTensor to_tensor = *to;
      to_tensor.device = RemoveRPCSessionMask(dev_to);
      to_tensor.data = static_cast<const RemoteSpace*>(to->data)->data;
      auto remote_dev = from_tensor.device;
      if (remote_dev.device_type == kDLCPU) remote_dev = to_tensor.device;
      GetSess(dev_from)->GetDeviceAPI(remote_dev)->CopyDataFromTo(&from_tensor, &to_tensor, stream);
    } else if (IsRPCSessionDevice(dev_from) && dev_to.device_type == kDLCPU) {
      DLTensor from_tensor = *from;
      from_tensor.device = RemoveRPCSessionMask(dev_from);
      from_tensor.data = static_cast<const RemoteSpace*>(from->data)->data;
      void* to_bytes = static_cast<char*>(to->data) + to->byte_offset;
      size_t nbytes = GetDataSize(*to);
      GetSess(dev_from)->CopyFromRemote(&from_tensor, to_bytes, nbytes);
    } else if (dev_from.device_type == kDLCPU && IsRPCSessionDevice(dev_to)) {
      DLTensor to_tensor = *to;
      to_tensor.device = RemoveRPCSessionMask(dev_to);
      to_tensor.data = static_cast<const RemoteSpace*>(to->data)->data;
      void* from_bytes = static_cast<char*>(from->data) + from->byte_offset;
      size_t nbytes = GetDataSize(*from);
      GetSess(dev_to)->CopyToRemote(from_bytes, &to_tensor, nbytes);
    } else {
      LOG(FATAL) << "expect copy from/to remote or between remote";
    }
  }

  TVMStreamHandle CreateStream(Device dev) {
    auto remote_dev = RemoveRPCSessionMask(dev);
    return GetSess(dev)->GetDeviceAPI(remote_dev)->CreateStream(remote_dev);
  }

  void FreeStream(Device dev, TVMStreamHandle stream) {
    auto remote_dev = RemoveRPCSessionMask(dev);
    GetSess(dev)->GetDeviceAPI(remote_dev)->FreeStream(remote_dev, stream);
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    auto remote_dev = RemoveRPCSessionMask(dev);
    GetSess(dev)->GetDeviceAPI(remote_dev)->StreamSync(remote_dev, stream);
  }

  void SetStream(Device dev, TVMStreamHandle stream) final {
    auto remote_dev = RemoveRPCSessionMask(dev);
    GetSess(dev)->GetDeviceAPI(remote_dev)->SetStream(remote_dev, stream);
  }

  TVMStreamHandle GetCurrentStream(Device dev) final {
    auto remote_dev = RemoveRPCSessionMask(dev);
    return GetSess(dev)->GetDeviceAPI(remote_dev)->GetCurrentStream(remote_dev);
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                      size_t num_bytes, Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    LOG(FATAL) << "Not implemented.";
  }

 private:
  std::shared_ptr<RPCSession> GetSess(Device dev) {
    int tbl_index = GetRPCSessionIndex(dev);
    return RPCSession::Get(tbl_index);
  }
};

TVM_REGISTER_GLOBAL("device_api.rpc").set_body([](TVMArgs args, TVMRetValue* rv) {
  static RPCDeviceAPI inst;
  DeviceAPI* ptr = &inst;
  *rv = static_cast<void*>(ptr);
});
}  // namespace runtime
}  // namespace tvm
