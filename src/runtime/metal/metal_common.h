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
 *  Copyright (c) 2017 by Contributors
 * \file metal_common.h
 * \brief Metal common header
 */
#ifndef TVM_RUNTIME_METAL_METAL_COMMON_H_
#define TVM_RUNTIME_METAL_METAL_COMMON_H_

#import <Metal/MTLBuffer.h>
#import <Metal/MTLCommandQueue.h>
#import <Metal/MTLCommandBuffer.h>
#import <Metal/MTLBlitCommandEncoder.h>
#import <Metal/MTLDevice.h>
#import <Metal/MTLLibrary.h>

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <dmlc/logging.h>
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include "../workspace_pool.h"

namespace tvm {
namespace runtime {
namespace metal {
/*!
 * \brief Process global Metal workspace.
 */
class MetalWorkspace final : public DeviceAPI {
 public:
  // the devices
  std::vector<id<MTLDevice> > devices;
  // the queues
  std::vector<id<MTLCommandQueue> > queues;
  // Warp size constant
  std::vector<int> warp_size;
  // Whether it is initialized.
  bool initialized_{false};
  // the mutex for initialization
  std::mutex mutex;
  // Destructor
  ~MetalWorkspace();
  // Get command queue for given context.
  id<MTLCommandQueue> GetCommandQueue(TVMContext ctx) {
    CHECK_EQ(ctx.device_type, kDLMetal);
    CHECK(ctx.device_id >= 0  && static_cast<size_t>(ctx.device_id) < queues.size())
        << "Invalid Metal device_id=" << ctx.device_id;
    return queues[ctx.device_id];
  }
  // Get device for given context
  id<MTLDevice> GetDevice(TVMContext ctx) {
    CHECK_EQ(ctx.device_type, kDLMetal);
    CHECK(ctx.device_id >= 0  && static_cast<size_t>(ctx.device_id) < devices.size())
        << "Invalid Metal device_id=" << ctx.device_id;
    return devices[ctx.device_id];
  }
  // Initialize workspace
  // Return false if already initialized, otherwise return true.
  void Init();
  // override device API
  void SetDevice(TVMContext ctx) final;
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final;
  void FreeDataSpace(TVMContext ctx, void* ptr) final;
  void CopyDataFromTo(const void* from,
                      size_t from_size,
                      void* to,
                      size_t to_size,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMType type_hint,
                      TVMStreamHandle stream) final;
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final;
  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final;
  void FreeWorkspace(TVMContext ctx, void* data) final;
  // get the global workspace
  static const std::shared_ptr<MetalWorkspace>& Global();
};

/*! \brief Thread local workspace */
class MetalThreadEntry {
 public:
  /*! \brief The current context */
  TVMContext context;
  /*! \brief The shared buffer used for copy. */
  std::vector<id<MTLBuffer> > temp_buffer_;
  /*! \brief workspace pool */
  WorkspacePool pool;
  // constructor
  MetalThreadEntry()
      : pool(static_cast<DLDeviceType>(kDLMetal), MetalWorkspace::Global()) {
    context.device_id = 0;
    context.device_type = static_cast<DLDeviceType>(kDLMetal);
  }
  ~MetalThreadEntry();
  // Get temp buffer with at least size under ctx.
  id<MTLBuffer> GetTempBuffer(TVMContext ctx, size_t size);
  // get the global workspace
  static MetalThreadEntry* ThreadLocal();
};
}  // namespace metal
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_METAL_METAL_COMMON_H_
