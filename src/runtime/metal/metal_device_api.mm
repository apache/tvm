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
 * \file metal_device_api.mm
 */
#include <tvm/runtime/registry.h>
#include <dmlc/thread_local.h>
#include "metal_common.h"

namespace tvm {
namespace runtime {
namespace metal {

const std::shared_ptr<MetalWorkspace>& MetalWorkspace::Global() {
  static std::shared_ptr<MetalWorkspace> inst =
      std::make_shared<MetalWorkspace>();
  return inst;
}

void MetalWorkspace::GetAttr(
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(ctx.device_id);
  if (kind == kExist) {
    *rv = int(index< devices.size());
    return;
  }
  CHECK_LT(index, devices.size())
      << "Invalid device id " << index;
  switch (kind) {
    case kMaxThreadsPerBlock: {
      *rv = static_cast<int>(
          [devices[ctx.device_id] maxThreadsPerThreadgroup].width);
      break;
    }
    case kWarpSize: {
      // Set warp size to be 1 for safty reason.
      *rv = 1;
      break;
    }
    case kMaxSharedMemoryPerBlock: return;
    case kComputeVersion: return;
    case kDeviceName: return;
    case kMaxClockRate: return;
    case kMultiProcessorCount: return;
    case kMaxThreadDimensions: return;
    case kExist: break;
  }
}

static const char* kDummyKernel = R"A0B0(
using namespace metal;
// Simple copy kernel
// Just to get threadExecutionWidth from current Metal API.
kernel void CopyKernel(
  device float* dst [[buffer(0)]],
  device float* src [[buffer(1)]],
  ushort2 gid[[thread_position_in_grid]]) {
  dst[gid.x] = src[gid.x];
}
)A0B0";

// Hack to get Warp size from device.
// Note that in Metal
// state.threadExecutionWidth can vary per kernel
// maybe due to resource constraint.
// so state.threadExecutionWidth can be smaller than warp size
// For safe issue, turn off warp-aware optimization for now
// But we keep this code.
int GetWarpSize(id<MTLDevice> dev) {
  NSError* error_msg = nil;
  id<MTLLibrary> lib =
      [dev
        newLibraryWithSource:
          [NSString stringWithUTF8String:kDummyKernel]
        options:nil
        error:&error_msg];
  CHECK(lib != nil) << [[error_msg localizedDescription] UTF8String];
  id<MTLFunction> f =
      [lib
        newFunctionWithName:
          [NSString stringWithUTF8String:"CopyKernel"]];
  CHECK(f!= nil);
  id<MTLComputePipelineState> state =
      [dev
        newComputePipelineStateWithFunction:f
        error:&error_msg];
  CHECK(state != nil) << [[error_msg localizedDescription] UTF8String];
  return static_cast<int>(state.threadExecutionWidth);
}

MetalWorkspace::~MetalWorkspace() {
  for (auto x : devices) {
    [x release];
  }
  for (auto x : queues) {
    [x release];
  }
}

void MetalWorkspace::Init() {
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mutex);
  if (initialized_) return;
  initialized_ = true;
  if (devices.size() != 0) return;
#if TARGET_OS_IPHONE
    // on iPhone
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    devices.push_back([d retain]);
    queues.push_back([[d newCommandQueue] retain]);
#else
    NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
    for (size_t i = 0; i < devs.count; ++i) {
      id<MTLDevice> d = [devs objectAtIndex:i];
      devices.push_back([d retain]);
      queues.push_back([[d newCommandQueue] retain]);
      LOG(INFO) << "Intializing Metal device " << i
                <<  ", name=" << [d.name UTF8String];
      warp_size.push_back(GetWarpSize(d));
    }
#endif
}

void MetalWorkspace::SetDevice(TVMContext ctx) {
  MetalThreadEntry::ThreadLocal()->context.device_id = ctx.device_id;
}

void* MetalWorkspace::AllocDataSpace(
    TVMContext ctx, size_t nbytes, size_t alignment, TVMType type_hint) {
  this->Init();
  id<MTLDevice> dev = GetDevice(ctx);
  // GPU memory only
  MTLResourceOptions storage_mode = MTLResourceStorageModePrivate;
  /*
  #if TARGET_OS_IPHONE
  storage_mode = MTLResourceStorageModeShared;
  #else
  storage_mode = MTLResourceStorageModeManaged;
  #endif
  */
  id<MTLBuffer> buf = [
      dev newBufferWithLength:nbytes
          options:storage_mode];
  CHECK(buf != nil);
  return (__bridge void*)([buf retain]);
}

void MetalWorkspace::FreeDataSpace(TVMContext ctx, void* ptr) {
  // release the ptr.
  CFRelease(ptr);
}

void MetalWorkspace::CopyDataFromTo(const void* from,
                                    size_t from_offset,
                                    void* to,
                                    size_t to_offset,
                                    size_t size,
                                    TVMContext ctx_from,
                                    TVMContext ctx_to,
                                    TVMType type_hint,
                                    TVMStreamHandle stream) {
  this->Init();
  CHECK(stream == nullptr);
  TVMContext ctx = ctx_from;
  if (ctx_from.device_type == kDLCPU) ctx = ctx_to;
  id<MTLCommandQueue> queue = GetCommandQueue(ctx);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  int from_dev_type = static_cast<int>(ctx_from.device_type);
  int to_dev_type = static_cast<int>(ctx_to.device_type);

  if (from_dev_type == kDLMetal && to_dev_type == kDLMetal) {
    CHECK_EQ(ctx_from.device_id, ctx_to.device_id)
        << "Metal disallow cross device copy.";
    id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
    [encoder copyFromBuffer:(__bridge id<MTLBuffer>)(from)
             sourceOffset:from_offset
             toBuffer:(__bridge id<MTLBuffer>)(to)
             destinationOffset:to_offset
             size:size];
    [encoder endEncoding];
    [cb commit];
  } else if (from_dev_type == kDLMetal && to_dev_type == kDLCPU) {
    // copy to a local buffer before get into global buffer.
    id<MTLBuffer> from_buf = (__bridge id<MTLBuffer>)(from);
    if (from_buf.storageMode != MTLStorageModeShared) {
      id<MTLBuffer> temp = MetalThreadEntry::ThreadLocal()
          ->GetTempBuffer(ctx_from, size);
      id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
      [encoder copyFromBuffer:from_buf
               sourceOffset:from_offset
               toBuffer:temp
               destinationOffset:0
               size:size];
      [encoder endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      memcpy(static_cast<char*>(to) + to_offset,
             static_cast<char*>([temp contents]),
             size);
    } else {
      memcpy(static_cast<char*>(to) + to_offset,
             static_cast<char*>([from_buf contents]) + from_offset,
             size);
    }
  } else if (from_dev_type == kDLCPU && to_dev_type == kDLMetal) {
    id<MTLBuffer> to_buf = (__bridge id<MTLBuffer>)(to);
    if (to_buf.storageMode != MTLStorageModeShared) {
      id<MTLBuffer> temp = MetalThreadEntry::ThreadLocal()
          ->GetTempBuffer(ctx_to, size);
      memcpy([temp contents],
              static_cast<const char*>(from) + from_offset,
              size);
      id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
      [encoder copyFromBuffer:temp
               sourceOffset:0
               toBuffer:to_buf
               destinationOffset:to_offset
               size:size];
      [encoder endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    } else {
      memcpy(static_cast<char*>([to_buf contents]) + to_offset,
             static_cast<const char*>(from) + from_offset,
             size);
    }
  } else {
    LOG(FATAL) << "Expect copy from/to Metal or between Metal"
               << ", from=" << from_dev_type
               << ", to=" << to_dev_type;
  }
}

void MetalWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  // commit an empty command buffer and wait until it completes.
  id<MTLCommandQueue> queue = GetCommandQueue(ctx);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  [cb commit];
  [cb waitUntilCompleted];
}

void* MetalWorkspace::AllocWorkspace(TVMContext ctx,
                                     size_t size,
                                     TVMType type_hint) {
  return MetalThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
}

void MetalWorkspace::FreeWorkspace(TVMContext ctx, void* data) {
  MetalThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
}

MetalThreadEntry::~MetalThreadEntry() {
  for (auto x : temp_buffer_) {
    if (x != nil) [x release];
  }
}

id<MTLBuffer> MetalThreadEntry::GetTempBuffer(TVMContext ctx, size_t size) {
  if (temp_buffer_.size() <= static_cast<size_t>(ctx.device_id)) {
    temp_buffer_.resize(ctx.device_id + 1, nil);
  }
  if (temp_buffer_[ctx.device_id] == nil ||
      temp_buffer_[ctx.device_id].length < size) {
    id<MTLDevice> dev = MetalWorkspace::Global()->GetDevice(ctx);
    if (temp_buffer_[ctx.device_id] != nil) {
      [temp_buffer_[ctx.device_id] release];
    }
    temp_buffer_[ctx.device_id] = [
        [dev newBufferWithLength:size
            options:MTLStorageModeShared] retain];
  }
  return temp_buffer_[ctx.device_id];
}

typedef dmlc::ThreadLocalStore<MetalThreadEntry> MetalThreadStore;

MetalThreadEntry* MetalThreadEntry::ThreadLocal() {
  return MetalThreadStore::Get();
}

TVM_REGISTER_GLOBAL("device_api.metal")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = MetalWorkspace::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace metal
}  // namespace runtime
}  // namespace tvm
