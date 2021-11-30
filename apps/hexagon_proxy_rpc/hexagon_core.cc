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

extern "C" {
#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt_error.h>
#include <qurt_hvx.h>
}

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <memory>
#include <string>

#include "common.h"
#include "hexagon_proxy_rpc.h"

template <typename T>
T* DeserializeToPointerType(unsigned int module) {
  return reinterpret_cast<T*>(module);
}

template <typename T>
unsigned int SerializeFromPointerType(T* pointer) {
  return *reinterpret_cast<unsigned int*>(&pointer);
}

tvm::runtime::Module load_module(const std::string& file_name) {
  static const tvm::runtime::PackedFunc loader =
      *tvm::runtime::Registry::Get("runtime.module.loadfile_hexagon");
  tvm::runtime::TVMRetValue rv = loader(file_name);
  if (rv.type_code() == kTVMModuleHandle) {
    return rv.operator tvm::runtime::Module();
  }
  return tvm::runtime::Module();
}

int __QAIC_HEADER(hexagon_proxy_rpc_open)(const char* uri, remote_handle64* handle) {
  FARF(ALWAYS, "[hexagon_proxy_rpc_open] FastRPC connection established");
  *handle = 0;
  const tvm::runtime::PackedFunc api = *tvm::runtime::Registry::Get("device_api.hexagon.v2");
  tvm::runtime::Registry::Register("device_api.hexagon", true).set_body(api);
  return AEE_SUCCESS;
}

int __QAIC_HEADER(hexagon_proxy_rpc_close)(remote_handle64 handle) {
  // Comment to stop clang-format from single-lining this function.
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(hexagon_proxy_rpc_load)(remote_handle64 handle, const char* module_path,
                                                unsigned int* module) {
  auto* mod_ptr = new tvm::runtime::Module(load_module(module_path));
  *module = SerializeFromPointerType<tvm::runtime::Module>(mod_ptr);
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(hexagon_proxy_rpc_unload)(remote_handle64 handle, unsigned int module) {
  tvm::runtime::Module* mod_ptr = DeserializeToPointerType<tvm::runtime::Module>(module);
  delete mod_ptr;
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(hexagon_proxy_rpc_get_function)(remote_handle64 handle, const char* name,
                                                        unsigned int module, unsigned int* func) {
  tvm::runtime::Module* mod_ptr = DeserializeToPointerType<tvm::runtime::Module>(module);
  std::string fname(name);
  tvm::runtime::PackedFunc f = (*mod_ptr)->GetFunction(fname);
  auto* f_ptr = new tvm::runtime::PackedFunc(f);
  *func = SerializeFromPointerType(f_ptr);
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(hexagon_proxy_rpc_release_function)(remote_handle64 handle,
                                                            unsigned int func) {
  tvm::runtime::PackedFunc* f_ptr = DeserializeToPointerType<tvm::runtime::PackedFunc>(func);
  delete f_ptr;
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(hexagon_proxy_rpc_invoke)(remote_handle64 handle, unsigned int func,
                                                  const unsigned char* handles, int nhandles) {
  tvm::runtime::PackedFunc* f_ptr = DeserializeToPointerType<tvm::runtime::PackedFunc>(func);
  const auto* meta = reinterpret_cast<const HandlePacket*>(handles);
  std::vector<TVMValue> values;
  std::vector<int> type_codes;
  for (size_t i = 0; i < meta->ndim; i++) {
    tvm::runtime::NDArray* array =
        DeserializeToPointerType<tvm::runtime::NDArray>(meta->handles[i]);
    type_codes.push_back(kTVMDLTensorHandle);
    values.emplace_back();
    const DLTensor* dltensor = array->operator->();
    values.back().v_handle = const_cast<void*>(static_cast<const void*>(dltensor));
  }

  {
    int res = qurt_hvx_reserve(QURT_HVX_RESERVE_ALL_AVAILABLE);
    switch (res) {
      case QURT_HVX_RESERVE_NOT_SUPPORTED:
      case QURT_HVX_RESERVE_NOT_SUCCESSFUL:
        FARF(ERROR, "error reserving HVX: %u", res);
        return AEE_EFAILED;
      default:
        break;
    }
    // Lock HVX.
    int lck = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (lck != 0) {
      FARF(ERROR, "error locking HVX: %u", lck);
      return AEE_EFAILED;
    }
  }
  tvm::runtime::TVMRetValue rv;
  f_ptr->CallPacked(tvm::runtime::TVMArgs(values.data(), type_codes.data(), values.size()), &rv);
  {
    int unl = qurt_hvx_unlock();
    if (unl != 0) {
      FARF(ERROR, "error unlocking HVX: %u", unl);
      return AEE_EFAILED;
    }
    // Release HVX.
    int rel = qurt_hvx_cancel_reserve();
    if (rel != 0) {
      FARF(ERROR, "error canceling HVX reservation: %u", rel);
      return AEE_EFAILED;
    }
  }

  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(hexagon_proxy_rpc_allocate)(remote_handle64 handle,
                                                    const unsigned char* input_meta,
                                                    int input_meta_size, const char* mem_scope,
                                                    unsigned int* tensor) {
  const auto* meta = reinterpret_cast<const tensor_meta*>(input_meta);
  auto device = tvm::Device{static_cast<DLDeviceType>(kDLHexagon), 0};
  tvm::runtime::Optional<tvm::runtime::String> scope;
  if (*mem_scope) {
    scope = mem_scope;
  }
  auto* array = new tvm::runtime::NDArray(std::move(tvm::runtime::NDArray::Empty(
      tvm::ShapeTuple(meta->shape, meta->shape + meta->ndim), meta->dtype, device, scope)));
  *tensor = SerializeFromPointerType(array);
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(hexagon_proxy_rpc_read)(remote_handle64 handle, unsigned char* dst_ptr,
                                                int nbytes, unsigned int src) {
  tvm::runtime::NDArray* src_ptr = DeserializeToPointerType<tvm::runtime::NDArray>(src);
  const DLTensor* t = src_ptr->operator->();
  tvm::ShapeTuple shape(t->shape, t->shape + t->ndim);
  auto* container = new tvm::runtime::NDArray::Container(
      static_cast<void*>(dst_ptr), shape, src_ptr->operator->()->dtype, tvm::Device{kDLCPU, 0});
  container->SetDeleter([](tvm::Object* container) {
    delete static_cast<tvm::runtime::NDArray::Container*>(container);
  });
  tvm::runtime::NDArray dst(GetObjectPtr<tvm::Object>(container));
  dst.CopyFrom(*src_ptr);
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(hexagon_proxy_rpc_write)(remote_handle64 handle, unsigned int dst,
                                                 const unsigned char* src_ptr, int nbytes) {
  tvm::runtime::NDArray* dst_ptr = DeserializeToPointerType<tvm::runtime::NDArray>(dst);
  const DLTensor* t = dst_ptr->operator->();
  tvm::ShapeTuple shape(t->shape, t->shape + t->ndim);
  auto* container =
      new tvm::runtime::NDArray::Container(const_cast<unsigned char*>(src_ptr), shape,
                                           dst_ptr->operator->()->dtype, tvm::Device{kDLCPU, 0});
  container->SetDeleter([](tvm::Object* container) {
    delete static_cast<tvm::runtime::NDArray::Container*>(container);
  });
  tvm::runtime::NDArray src(GetObjectPtr<tvm::Object>(container));
  dst_ptr->CopyFrom(src);
  return AEE_SUCCESS;
}

AEEResult __QAIC_HEADER(hexagon_proxy_rpc_release)(remote_handle64 handle, unsigned int array) {
  tvm::runtime::NDArray* array_ptr = DeserializeToPointerType<tvm::runtime::NDArray>(array);
  delete array_ptr;
  return AEE_SUCCESS;
}
