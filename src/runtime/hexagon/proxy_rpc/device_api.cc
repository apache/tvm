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
 * \file device_api.cc
 * \brief Shim RPC Device API that forwards to/from Hexagon
 * over FastRPC.
 */

#if !defined(__ANDROID__)
#error HexagonRPCDeviceAPI is meant only for compilation on Android.
#endif

#include <rpcmem.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "../hexagon/hexagon_common.h"

namespace tvm {
namespace runtime {

class MirroredBuffer {
 public:
  MirroredBuffer(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                 Optional<String> scope) {
    DLTensor t;
    t.shape = const_cast<int64_t*>(shape);
    t.ndim = ndim;
    t.dtype = dtype;
    t.device = dev;
    rpc_mem_size_ = GetDataSize(t);
    rpc_mem_ = reinterpret_cast<uint8_t*>(
        rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, rpc_mem_size_));

    TVMValue value;
    value.v_handle = &t;
    TVMArgValue mem_scope;
    if (scope.defined()) {
      TVMValue s;
      s.v_str = scope.value().c_str();
      mem_scope = TVMArgValue(s, kTVMStr);
    }

    auto* f = runtime::Registry::Get("tvm.rpc.hexagon.allocate");
    hexagon_buffer_id_ = (*f)(TVMArgValue(value, kTVMDLTensorHandle), mem_scope);
  }
  void Read() {
    auto* f = runtime::Registry::Get("tvm.rpc.hexagon.read_to_host");
    (*f)(rpc_mem_, rpc_mem_size_, hexagon_buffer_id_);
  }
  void Write() {
    auto* f = runtime::Registry::Get("tvm.rpc.hexagon.write_from_host");
    (*f)(hexagon_buffer_id_, rpc_mem_, rpc_mem_size_);
  }
  ~MirroredBuffer() {
    auto* f = runtime::Registry::Get("tvm.rpc.hexagon.release");
    (*f)(hexagon_buffer_id_);
    rpcmem_free(rpc_mem_);
  }
  void* GetRPCMem() { return rpc_mem_; }
  int32_t GetHexagonHandle() { return hexagon_buffer_id_; }

 private:
  void* rpc_mem_ = nullptr;
  size_t rpc_mem_size_ = 0;
  int32_t hexagon_buffer_id_ = 0;
};

class HexagonRPCDeviceAPI final : public DeviceAPI {
 public:
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    throw tvm::runtime::Error("HexagonRPCDeviceAPI::AllocDataSpace is unimplemented");
  }
  void* AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                       Optional<String> mem_scope) final {
    return new MirroredBuffer(dev, ndim, shape, dtype, mem_scope);
  }
  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final {
    ICHECK((IsHexagonDevice(from->device) && IsHexagonDevice(to->device)) == false)
        << "Unimplimented";
    if (IsHexagonDevice(from->device) && to->device.device_type == kDLCPU) {
      MirroredBuffer* mirror = static_cast<MirroredBuffer*>(from->data);
      mirror->Read();
      memcpy(static_cast<char*>(to->data) + to->byte_offset,
             static_cast<const char*>(mirror->GetRPCMem()) + from->byte_offset, GetDataSize(*from));
    } else if (from->device.device_type == kDLCPU && IsHexagonDevice(to->device)) {
      MirroredBuffer* mirror = static_cast<MirroredBuffer*>(to->data);
      memcpy(static_cast<char*>(mirror->GetRPCMem()) + to->byte_offset,
             static_cast<const char*>(from->data) + from->byte_offset, GetDataSize(*from));
      mirror->Write();
    } else {
      CHECK(false) << "Expect copy between DLTensor devices of types kDLHexagon and kDLCPU only.";
    }
  }
  void FreeDataSpace(Device dev, void* ptr) final {
    MirroredBuffer* mirror = static_cast<MirroredBuffer*>(ptr);
    delete mirror;
  }

  void SetDevice(Device dev) final {}
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void StreamSync(Device dev, TVMStreamHandle stream) final {}
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    throw tvm::runtime::Error("HexagonRPCDeviceAPI::AllocWorkspace is unimplemented");
  };
  void FreeWorkspace(Device dev, void* data) final {
    throw tvm::runtime::Error("HexagonRPCDeviceAPI::FreeWorkspace is unimplemented");
  };

  static HexagonRPCDeviceAPI* Global() {
    static auto* inst = new HexagonRPCDeviceAPI();
    return inst;
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    throw tvm::runtime::Error("HexagonRPCDeviceAPI::CopyDataFromTo is unimplemented");
  }
};

TVM_REGISTER_GLOBAL("runtime.hexagon.GetHandle").set_body([](TVMArgs args, TVMRetValue* rv) {
  auto* buf = static_cast<MirroredBuffer*>(static_cast<void*>(args[0]));
  *rv = buf->GetHexagonHandle();
});

TVM_REGISTER_GLOBAL("device_api.hexagon").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = HexagonRPCDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace runtime
}  // namespace tvm
