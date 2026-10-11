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

/*! \file rpc_remote.h
 *  \brief Session-owned RPC values. Remote addresses never enter local tensor storage.
 */
#ifndef TVM_RUNTIME_RPC_RPC_REMOTE_H_
#define TVM_RUNTIME_RPC_RPC_REMOTE_H_

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/reflection/registry.h>

#include <memory>
#include <utility>

#include "rpc_session.h"

namespace tvm {
namespace runtime {

class RemoteDeviceObj : public ffi::Object {
 public:
  std::shared_ptr<RPCSession> sess;
  Device endpoint_device;
  RPCSession* session() const {
    sess->EnsureOpen();
    return sess.get();
  }
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("rpc.RemoteDevice", RemoteDeviceObj, ffi::Object);
};
class RemoteDevice : public ffi::ObjectRef {
 public:
  explicit RemoteDevice(ffi::ObjectPtr<RemoteDeviceObj> data) : ffi::ObjectRef(std::move(data)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RemoteDevice, ffi::ObjectRef, RemoteDeviceObj);
};
inline RemoteDevice MakeRemoteDevice(std::shared_ptr<RPCSession> sess, Device dev) {
  sess->EnsureOpen();
  auto n = ffi::make_object<RemoteDeviceObj>();
  n->sess = std::move(sess);
  n->endpoint_device = dev;
  return RemoteDevice(n);
}

/*! \brief Shared ownership of a remote allocation or returned tensor object. */
struct RemoteTensorStorage {
  std::shared_ptr<RPCSession> sess;
  Device device;
  void* data{nullptr};
  void* object_handle{nullptr};
  bool owns_allocation{false};
  ~RemoteTensorStorage() {
    try {
      if (object_handle != nullptr) {
        sess->FreeHandle(object_handle);
      } else if (owns_allocation) {
        sess->EnsureOpen();
        sess->FreeTensor(device, data);
      }
    } catch (const ffi::Error&) {
      // A closed transport has already released its remote resources.
    }
  }
};

class RemoteTensorObj : public ffi::Object {
 public:
  RemoteDevice device{ffi::UnsafeInit{}};
  ffi::Shape shape;
  DLDataType dtype;
  uint64_t byte_offset{0};
  std::shared_ptr<RemoteTensorStorage> storage;
  /*! \brief Temporary descriptor for RPC transport only, never a local tensor. */
  DLTensor descriptor() const {
    device->sess->EnsureOpen();
    return {storage->data,
            device->endpoint_device,
            static_cast<int>(shape.size()),
            dtype,
            const_cast<int64_t*>(shape.data()),
            nullptr,
            byte_offset};
  }
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("rpc.RemoteTensor", RemoteTensorObj, ffi::Object);
};
class RemoteTensor : public ffi::ObjectRef {
 public:
  explicit RemoteTensor(ffi::ObjectPtr<RemoteTensorObj> data) : ffi::ObjectRef(std::move(data)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RemoteTensor, ffi::ObjectRef, RemoteTensorObj);
};

inline RemoteTensor RemoteTensorFromHandle(std::shared_ptr<RPCSession> sess, const DLTensor& tensor,
                                           void* object_handle) {
  auto n = ffi::make_object<RemoteTensorObj>();
  n->device = MakeRemoteDevice(sess, tensor.device);
  n->shape = ffi::Shape(tensor.shape, tensor.shape + tensor.ndim);
  n->dtype = tensor.dtype;
  n->byte_offset = tensor.byte_offset;
  n->storage = std::make_shared<RemoteTensorStorage>();
  n->storage->sess = std::move(sess);
  n->storage->device = tensor.device;
  n->storage->data = tensor.data;
  n->storage->object_handle = object_handle;
  TVM_FFI_CHECK(ffi::IsContiguous(tensor), ValueError)
      << "RPC requires a contiguous returned tensor";
  return RemoteTensor(n);
}

class RemoteStreamObj : public ffi::Object {
 public:
  RemoteDevice device{ffi::UnsafeInit{}};
  std::shared_ptr<RPCStreamState> state;
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("rpc.RemoteStream", RemoteStreamObj, ffi::Object);
};
class RemoteStream : public ffi::ObjectRef {
 public:
  explicit RemoteStream(ffi::ObjectPtr<RemoteStreamObj> data) : ffi::ObjectRef(std::move(data)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RemoteStream, ffi::ObjectRef, RemoteStreamObj);
};

inline TVMStreamHandle CheckRemoteStream(const RemoteDevice& dev,
                                         const ffi::Optional<RemoteStream>& stream) {
  dev->sess->EnsureOpen();
  if (!stream.has_value()) return nullptr;
  const auto& s = stream.value();
  TVM_FFI_CHECK(s->device->sess == dev->sess &&
                    s->device->endpoint_device.device_type == dev->endpoint_device.device_type &&
                    s->device->endpoint_device.device_id == dev->endpoint_device.device_id,
                ValueError)
      << "Stream belongs to a different remote device or session";
  TVM_FFI_CHECK(!s->state->closed, ValueError) << "Remote stream is closed";
  return s->state->handle;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_REMOTE_H_
