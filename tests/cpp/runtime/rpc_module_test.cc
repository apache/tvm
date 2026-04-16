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

#include <gtest/gtest.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <memory>
#include <string>

#include "../../../src/runtime/rpc/rpc_session.h"

namespace tvm {
namespace runtime {

Tensor TensorFromRemoteOpaqueHandle(std::shared_ptr<RPCSession> sess, void* handle,
                                    DLTensor* template_tensor, Device dev,
                                    void* remote_tensor_handle);

namespace {

class RecordingRPCSession final : public RPCSession {
 public:
  PackedFuncHandle GetFunction(const std::string& name) final { return nullptr; }

  void CallFunc(PackedFuncHandle func, ffi::PackedArgs args,
                const FEncodeReturn& fencode_return) final {}

  void CopyToRemote(void* local_from_bytes, DLTensor* remote_to, uint64_t nbytes) final {}

  void CopyFromRemote(DLTensor* remote_from, void* local_to_bytes, uint64_t nbytes) final {}

  void FreeHandle(void* handle) final {
    ++free_handle_calls;
    last_freed_handle = handle;
    if (throw_on_free) {
      TVM_FFI_THROW(InternalError) << "simulated remote close";
    }
  }

  DeviceAPI* GetDeviceAPI(Device dev, bool allow_missing = false) final { return nullptr; }

  bool IsLocalSession() const final { return false; }

  int free_handle_calls{0};
  void* last_freed_handle{nullptr};
  bool throw_on_free{false};
};

DLTensor MakeTemplateTensor() {
  static int64_t shape[1] = {4};
  DLTensor tensor{};
  tensor.data = nullptr;
  tensor.device = Device{kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = DataType::Float(32);
  tensor.shape = shape;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  return tensor;
}

Device MakeRemoteDevice(const std::shared_ptr<RPCSession>& sess) {
  return AddRPCSessionMask(Device{kDLCPU, 0}, sess->table_index());
}

}  // namespace

TEST(RPCTensorTest, ReturnedTensorFreesRemoteTensorHandle) {
  auto sess = std::make_shared<RecordingRPCSession>();
  DLTensor template_tensor = MakeTemplateTensor();
  void* data_handle = reinterpret_cast<void*>(0x1234);
  void* tensor_handle = reinterpret_cast<void*>(0x5678);

  {
    auto tensor = TensorFromRemoteOpaqueHandle(sess, data_handle, &template_tensor,
                                               MakeRemoteDevice(sess), tensor_handle);
    EXPECT_NE(tensor.defined(), false);
  }

  EXPECT_EQ(sess->free_handle_calls, 1);
  EXPECT_EQ(sess->last_freed_handle, tensor_handle);
  EXPECT_NE(sess->last_freed_handle, data_handle);
}

TEST(RPCTensorTest, ReturnedTensorDestructorIgnoresFreeHandleErrors) {
  auto sess = std::make_shared<RecordingRPCSession>();
  sess->throw_on_free = true;
  DLTensor template_tensor = MakeTemplateTensor();
  void* data_handle = reinterpret_cast<void*>(0x1234);
  void* tensor_handle = reinterpret_cast<void*>(0x5678);

  EXPECT_NO_THROW({
    auto tensor = TensorFromRemoteOpaqueHandle(sess, data_handle, &template_tensor,
                                               MakeRemoteDevice(sess), tensor_handle);
  });
  EXPECT_EQ(sess->free_handle_calls, 1);
  EXPECT_EQ(sess->last_freed_handle, tensor_handle);
}

}  // namespace runtime
}  // namespace tvm
