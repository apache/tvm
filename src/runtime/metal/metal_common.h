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
#include <utility>
#include <vector>

#include "../workspace_pool.h"

/* Macro for convenience in using AutoReleasePoolWrapper.
 * With this macro we can add AutoReleasePoolWrapper to our ObjC code in more
 * native way.
 *
 * For example, this is ObjC code with autoreleasepool:
 *     @autoreleasepool {
 *         // Some code
 *     }
 *
 * To avoid possible memory leaks when an exception will be generated, we
 * should update this code:
 *     AUTORELEASEPOOL { // Replace @autoreleasepool -> AUTORELEASEPOOL
 *         // Some code
 *     }; // Add semicolon after close bracket
 *
 * In macro AUTORELEASEPOOL we get the instance of AutoReleasePoolWrapper and
 * put a lambda function with code from autoreleasepool to the insertion
 * operator of AutoReleasePoolWrapper class.
 *
 * Note: If you want to return a value from the autoreleasepool, you should
 * declare the variable with result before AUTORELEASEPOOL macro. This variable
 * will be captured by reference and you can use it in the code in autorelease
 * pool. But you should write return statement after AUTORELEASEPOOL macro.
 */
#define AUTORELEASEPOOL tvm::runtime::metal::AutoReleasePoolWrapper::GetInstance() << [&]()

namespace tvm {
namespace runtime {
namespace metal {
/*!
 * \brief Wrapper on autoreleasepool with exception handling
 *
 * \note In case when the exception was thrown from the autoreleasepool, the
 * allocated resources won't be released in proper way. So, we handle exception
 * in autoreleasepool and after the autoreleasepool we rethrow this exception.
 */
class AutoReleasePoolWrapper {
 public:
  static AutoReleasePoolWrapper& GetInstance();
  template <typename T>
  void operator<<(const T& f) {
    std::exception_ptr eptr;
    @autoreleasepool {
      try {
        f();
      } catch (...) {
        eptr = std::current_exception();
      }
    }
    if (eptr) std::rethrow_exception(eptr);
  }

 private:
  AutoReleasePoolWrapper() = default;
  ~AutoReleasePoolWrapper() = default;
  AutoReleasePoolWrapper(const AutoReleasePoolWrapper&) = delete;
  AutoReleasePoolWrapper& operator=(const AutoReleasePoolWrapper&) = delete;
};

/*!
 * \brief Structure for error handling in queues
 */
class Stream {
 public:
  explicit Stream(id<MTLDevice> device) { queue_ = [device newCommandQueue]; }
  ~Stream() { [queue_ release]; }
  id<MTLCommandBuffer> GetCommandBuffer(std::string label = "", bool attach_error_callback = true) {
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    if (!label.empty()) {
      cb.label = [NSString stringWithUTF8String:label.c_str()];
    }
    [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
      if (buffer.status == MTLCommandBufferStatusError) {
        ICHECK(buffer.error != nil);
        this->SetError(buffer.error.localizedDescription.UTF8String);
      }
    }];
    return cb;
  }

  void SetError(std::string error_description) {
    error_happened_ = true;
    error_description_ = std::move(error_description);
  }

  bool HasErrorHappened() const { return error_happened_; }

  const std::string& ErrorDescription() const { return error_description_; }

 private:
  // Queue
  id<MTLCommandQueue> queue_;
  // Check if error happened in one previous run
  bool error_happened_{false};
  // error description
  std::string error_description_;
};

/*!
 * \brief Process global Metal workspace.
 */
class MetalWorkspace final : public DeviceAPI {
 public:
  // the devices
  std::vector<id<MTLDevice>> devices;
  // Warp size constant
  std::vector<int> warp_size;
  MetalWorkspace();
  // Destructor
  ~MetalWorkspace();
  // Get device for given device
  id<MTLDevice> GetDevice(Device dev) {
    ICHECK_EQ(dev.device_type, kDLMetal);
    ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < devices.size())
        << "Invalid Metal device_id=" << dev.device_id;
    return devices[dev.device_id];
  }
  // override device API
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  TVMStreamHandle CreateStream(Device dev) final;
  void FreeStream(Device dev, TVMStreamHandle stream) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;
  void SetStream(Device dev, TVMStreamHandle stream) final;
  TVMStreamHandle GetCurrentStream(Device dev) final;
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;
  void ReinitializeDefaultStreams();

  /**
   * Cast stream to the right metal stream data structure
   * if stream is nullptr , return the default stream of device_id
   * \param stream the input stream handle
   * \param device_id The device id of interest
   * \returns The stream used in this function.
   */
  Stream* CastStreamOrGetDefault(TVMStreamHandle stream, int device_id);

  // get the global workspace
  static MetalWorkspace* Global();

 protected:
  void CopyDataFromTo(const void* from, size_t from_size, void* to, size_t to_size, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;

 private:
  // Pointers to default allocated streams
  std::vector<Stream*> default_streams_;
};

/*! \brief Thread local workspace */
class MetalThreadEntry {
 public:
  /*! \brief The current device */
  Device device;
  /*! \brief The current stream */
  std::vector<TVMStreamHandle> stream;
  /*! \brief The shared buffer used for copy. */
  std::vector<id<MTLBuffer>> temp_buffer_;
  /*! \brief workspace pool */
  WorkspacePool pool;
  // constructor
  MetalThreadEntry() : pool(static_cast<DLDeviceType>(kDLMetal), MetalWorkspace::Global()) {
    device.device_id = 0;
    device.device_type = static_cast<DLDeviceType>(kDLMetal);
    MetalWorkspace* global_ws = MetalWorkspace::Global();
    // by default, set the stream to nullptr, which indicate
    // that we are using default stream
    this->stream.resize(global_ws->devices.size(), nullptr);
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
