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
 * \file metal_common.h
 * \brief Metal common header
 */
#ifndef TVM_RUNTIME_METAL_METAL_COMMON_H_
#define TVM_RUNTIME_METAL_METAL_COMMON_H_

#import <Metal/MTLBlitCommandEncoder.h>
#import <Metal/MTLBuffer.h>
#import <Metal/MTLCommandBuffer.h>
#import <Metal/MTLCommandQueue.h>
#import <Metal/MTLDevice.h>
#import <Metal/MTLLibrary.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

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
  // Get command queue for given device.
  id<MTLCommandQueue> GetCommandQueue(Device dev) {
    ICHECK_EQ(dev.device_type, kDLMetal);
    ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < queues.size())
        << "Invalid Metal device_id=" << dev.device_id;
    return queues[dev.device_id];
  }
  // Get device for given device
  id<MTLDevice> GetDevice(Device dev) {
    ICHECK_EQ(dev.device_type, kDLMetal);
    ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < devices.size())
        << "Invalid Metal device_id=" << dev.device_id;
    return devices[dev.device_id];
  }
  // Initialize workspace
  // Return false if already initialized, otherwise return true.
  void Init();
  // override device API
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;
  // get the global workspace
  static MetalWorkspace* Global();

 protected:
  void CopyDataFromTo(const void* from, size_t from_size, void* to, size_t to_size, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;
};

/*! \brief Thread local workspace */
class MetalThreadEntry {
 public:
  /*! \brief The current device */
  Device device;
  /*! \brief The shared buffer used for copy. */
  std::vector<id<MTLBuffer> > temp_buffer_;
  /*! \brief workspace pool */
  WorkspacePool pool;
  // constructor
  MetalThreadEntry() : pool(static_cast<DLDeviceType>(kDLMetal), MetalWorkspace::Global()) {
    device.device_id = 0;
    device.device_type = static_cast<DLDeviceType>(kDLMetal);
  }
  ~MetalThreadEntry();
  // Get temp buffer with at least size under dev.
  id<MTLBuffer> GetTempBuffer(Device dev, size_t size);
  // get the global workspace
  static MetalThreadEntry* ThreadLocal();
};
}  // namespace metal
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_METAL_METAL_COMMON_H_
