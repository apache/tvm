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
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>

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
 * \brief Metal command stream with batched dispatch support.
 *
 * Compute dispatches are batched into a single command buffer via
 * GetPendingComputeEncoder(). Blit operations (copies) are interleaved
 * on the same command buffer via GetBlitEncoderOnPendingBuffer().
 * The command buffer is committed when FlushCommandBuffer() is called.
 *
 * Must call FlushCommandBuffer() before:
 * - GPU→CPU readback (need data in CPU memory)
 * - Buffer deallocation (FreeDataSpace, setPurgeableState:Empty on
 *   a buffer referenced by an uncommitted CB crashes Metal)
 * - Stream sync (StreamSync / Synchronize)
 */
class Stream {
 public:
  explicit Stream(id<MTLDevice> device) { queue_ = [device newCommandQueue]; }
  ~Stream() {
    FlushCommandBuffer();
    [queue_ release];
  }

  /*!
   * \brief Get a standalone command buffer (for GPU→CPU readback only).
   *
   * Used when we need a separate command buffer that we can commit
   * and waitUntilCompleted on independently.
   */
  id<MTLCommandBuffer> GetCommandBuffer(std::string label = "") {
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    if (!label.empty()) {
      cb.label = [NSString stringWithUTF8String:label.c_str()];
    }
    [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
      if (buffer.status == MTLCommandBufferStatusError) {
        TVM_FFI_ICHECK(buffer.error != nil);
        this->SetError(buffer.error.localizedDescription.UTF8String);
      }
    }];
    return cb;
  }

  /*!
   * \brief Get the pending compute command encoder, creating one if needed.
   *
   * Multiple compute dispatches are batched into a single command buffer
   * and encoder. Blit operations (copies) can be interleaved on the same
   * command buffer via GetBlitEncoderOnPendingBuffer(). The entire command
   * buffer is committed when FlushCommandBuffer() is called.
   *
   * Must flush before:
   * - GPU→CPU readback (need data on CPU immediately)
   * - Buffer deallocation (FreeDataSpace)
   * - Stream sync (StreamSync)
   */
  id<MTLComputeCommandEncoder> GetPendingComputeEncoder(const std::string& kernel_name = "") {
    if (pending_compute_encoder_ == nil) {
      id<MTLCommandBuffer> cb = GetOrCreatePendingCommandBuffer();
      pending_compute_encoder_ = [[cb computeCommandEncoder] retain];
    }
    if (!kernel_name.empty()) {
      last_dispatched_kernel_ = kernel_name;
    }
    profile.dispatches++;
    return pending_compute_encoder_;
  }

  /*!
   * \brief Get a blit encoder on the pending command buffer.
   *
   * Pauses the active compute encoder (if any), creates a blit encoder
   * on the same command buffer. Caller must call [encoder endEncoding]
   * when done. The next GetPendingComputeEncoder() call will create a
   * new compute encoder on the same command buffer.
   *
   * Metal guarantees sequential ordering of encoders within a command
   * buffer, so blits encoded here execute after prior compute dispatches
   * and before subsequent ones.
   */
  id<MTLBlitCommandEncoder> GetBlitEncoderOnPendingBuffer() {
    PauseComputeEncoder();
    id<MTLCommandBuffer> cb = GetOrCreatePendingCommandBuffer();
    profile.blits++;
    return [cb blitCommandEncoder];
  }

  /*!
   * \brief Flush: end active encoder, commit the command buffer.
   *
   * Safe to call when nothing is pending (no-op).
   */
  void FlushCommandBuffer() {
    PauseComputeEncoder();
    if (pending_command_buffer_ != nil) {
      [pending_command_buffer_ commit];
      [pending_command_buffer_ release];
      pending_command_buffer_ = nil;
      profile.flushes++;
    }
  }

  /*!
   * \brief Flush pending work, then wait for all submitted work to complete.
   */
  void Synchronize() {
    FlushCommandBuffer();
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
      if (buffer.status == MTLCommandBufferStatusError) {
        TVM_FFI_ICHECK(buffer.error != nil);
        this->SetError(buffer.error.localizedDescription.UTF8String);
      }
    }];
    [cb commit];
    [cb waitUntilCompleted];
    profile.syncs++;
  }

  bool HasPendingWork() const { return pending_command_buffer_ != nil; }

  /*! \brief Profiling counters for diagnosing dispatch/copy/sync overhead. */
  struct ProfileCounters {
    size_t dispatches = 0;
    size_t flushes = 0;
    size_t syncs = 0;
    size_t blits = 0;
    size_t gpu_to_cpu = 0;
    size_t cpu_to_gpu = 0;
    size_t gpu_to_gpu = 0;
    size_t free_syncs = 0;  // FreeDataSpace calls that triggered a sync

    void Reset() { *this = ProfileCounters(); }
  };
  ProfileCounters profile;

  void SetError(std::string error_description) {
    error_happened_ = true;
    error_description_ = std::move(error_description);
  }

  bool HasErrorHappened() const { return error_happened_; }

  const std::string& ErrorDescription() const { return error_description_; }

 private:
  /*! \brief Get or create the pending command buffer (shared by compute and blit). */
  id<MTLCommandBuffer> GetOrCreatePendingCommandBuffer() {
    if (pending_command_buffer_ == nil) {
      pending_command_buffer_ = [[queue_ commandBuffer] retain];
      pending_command_buffer_.label = @"TVMBatched";
      [pending_command_buffer_ addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        if (buffer.status == MTLCommandBufferStatusError) {
          TVM_FFI_ICHECK(buffer.error != nil);
          std::string msg = buffer.error.localizedDescription.UTF8String;
          if (!this->last_dispatched_kernel_.empty()) {
            msg = "GPUError after kernel " + this->last_dispatched_kernel_ + ": " + msg;
          }
          this->SetError(msg);
        }
      }];
    }
    return pending_command_buffer_;
  }

  /*! \brief End the active compute encoder without committing the command buffer. */
  void PauseComputeEncoder() {
    if (pending_compute_encoder_ != nil) {
      [pending_compute_encoder_ endEncoding];
      [pending_compute_encoder_ release];
      pending_compute_encoder_ = nil;
    }
  }

  // Queue
  id<MTLCommandQueue> queue_;
  // Pending command buffer (shared by compute and blit encoders)
  id<MTLCommandBuffer> pending_command_buffer_ = nil;
  // Active compute encoder on the pending command buffer (nil when paused/blit)
  id<MTLComputeCommandEncoder> pending_compute_encoder_ = nil;
  // Last dispatched kernel name (for error diagnostics)
  std::string last_dispatched_kernel_;
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
    TVM_FFI_ICHECK_EQ(dev.device_type, kDLMetal);
    TVM_FFI_ICHECK(dev.device_id >= 0 && static_cast<size_t>(dev.device_id) < devices.size())
        << "Invalid Metal device_id=" << dev.device_id;
    return devices[dev.device_id];
  }
  // override device API
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final;
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  TVMStreamHandle CreateStream(Device dev) final;
  void FreeStream(Device dev, TVMStreamHandle stream) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;
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
  /*! \brief The shared buffer used for GPU→CPU readback. */
  std::vector<id<MTLBuffer>> temp_buffer_;
  /*!
   * \brief Pool of staging buffers for CPU→GPU copies that are inlined
   * into the pending command buffer. Each inlined copy needs its own
   * staging buffer because the GPU reads them asynchronously.
   * Buffers are recycled after FlushCommandBuffer()/Synchronize().
   */
  struct StagingBufferPool {
    struct Entry {
      id<MTLBuffer> buffer = nil;
      size_t size = 0;
    };
    std::vector<Entry> pool;
    size_t next_index = 0;  // sequential within current batch, reset on sync

    id<MTLBuffer> GetOrCreate(id<MTLDevice> dev, size_t nbytes) {
      if (next_index < pool.size() && pool[next_index].size >= nbytes) {
        return pool[next_index++].buffer;
      }
      // Need a new or bigger buffer at this index
      if (next_index < pool.size() && pool[next_index].buffer != nil) {
        [pool[next_index].buffer release];
      }
      if (next_index >= pool.size()) {
        pool.push_back({nil, 0});
      }
      pool[next_index].buffer = [dev newBufferWithLength:nbytes options:MTLStorageModeShared];
      TVM_FFI_ICHECK(pool[next_index].buffer != nil)
          << "Failed to allocate staging buffer of size " << nbytes;
      pool[next_index].size = nbytes;
      return pool[next_index++].buffer;
    }

    // Called after flush/sync, all staging buffers are safe to reuse
    void ResetIndex() { next_index = 0; }

    ~StagingBufferPool() {
      for (auto& e : pool) {
        if (e.buffer != nil) {
          [e.buffer release];
        }
      }
    }
  };
  std::vector<StagingBufferPool> staging_pools_;  // per device
  /*! \brief workspace pool */
  WorkspacePool pool;
  // constructor
  MetalThreadEntry() : pool(static_cast<DLDeviceType>(kDLMetal), MetalWorkspace::Global()) {
    device.device_id = 0;
    device.device_type = static_cast<DLDeviceType>(kDLMetal);
    MetalWorkspace* global_ws = MetalWorkspace::Global();
    this->stream.resize(global_ws->devices.size(), nullptr);
    this->staging_pools_.resize(global_ws->devices.size());
  }
  ~MetalThreadEntry();
  // Get temp buffer with at least size under dev (for GPU→CPU readback).
  id<MTLBuffer> GetTempBuffer(Device dev, size_t size);
  // Get a staging buffer for inlined CPU→GPU copy (from pool).
  id<MTLBuffer> GetOrCreateStagingBuffer(Device dev, size_t size);
  // get the global workspace
  static MetalThreadEntry* ThreadLocal();
};
}  // namespace metal
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_METAL_METAL_COMMON_H_
