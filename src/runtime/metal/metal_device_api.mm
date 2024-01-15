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
 * \file metal_device_api.mm
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include "metal_common.h"

namespace tvm {
namespace runtime {
namespace metal {

AutoReleasePoolWrapper& AutoReleasePoolWrapper::GetInstance() {
  static AutoReleasePoolWrapper instance;
  return instance;
}

MetalWorkspace* MetalWorkspace::Global() {
  // NOTE: explicitly use new to avoid exit-time destruction of global state
  // Global state will be recycled by OS as the process exits.
  static MetalWorkspace* inst = new MetalWorkspace();
  return inst;
}

void MetalWorkspace::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  AUTORELEASEPOOL {
    this->Init();
    size_t index = static_cast<size_t>(dev.device_id);
    if (kind == kExist) {
      *rv = int(index < devices.size());
      return;
    }
    ICHECK_LT(index, devices.size()) << "Invalid device id " << index;
    switch (kind) {
      case kMaxThreadsPerBlock: {
        *rv = static_cast<int>([devices[dev.device_id] maxThreadsPerThreadgroup].width);
        break;
      }
      case kWarpSize: {
#if defined(__x86_64__)
        *rv = 1;
#elif defined(__aarch64__)
        *rv = 32;
#else
        LOG(WARNING) << "The CPU architecture is neither x86 nor aarch64. Fallback to warp size 1.";
        *rv = 1;
#endif
        break;
      }
      case kMaxSharedMemoryPerBlock:
        return;
      case kComputeVersion:
        return;
      case kDeviceName:
        return;
      case kMaxClockRate:
        return;
      case kMultiProcessorCount:
        return;
      case kMaxThreadDimensions:
        return;
      case kExist:
        return;
      case kMaxRegistersPerBlock:
        return;
      case kGcnArch:
        return;
      case kApiVersion:
        return;
      case kDriverVersion:
        return;
      case kL2CacheSizeBytes:
        return;
      case kTotalGlobalMemory: {
        *rv = static_cast<int64_t>([devices[dev.device_id] recommendedMaxWorkingSetSize]);
        return;
      }
    }
  };
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
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kDummyKernel]
                                         options:nil
                                           error:&error_msg];
  ICHECK(lib != nil) << [[error_msg localizedDescription] UTF8String];
  id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:"CopyKernel"]];
  ICHECK(f != nil);
  id<MTLComputePipelineState> state = [dev newComputePipelineStateWithFunction:f error:&error_msg];
  ICHECK(state != nil) << [[error_msg localizedDescription] UTF8String];
  int size = static_cast<int>(state.threadExecutionWidth);
  [state release];
  [f release];
  [lib release];
  return size;
}

MetalWorkspace::~MetalWorkspace() {
  for (auto x : devices) {
    [x release];
  }
  for (auto x : default_streams_) {
    delete x;
  }
}

void MetalWorkspace::ReinitializeStreams() {
  std::vector<Stream*>& threadStreams = MetalThreadEntry::ThreadLocal()->stream;
  ICHECK_EQ(default_streams_.size(), threadStreams.size());
  for (size_t i = 0; i < default_streams_.size(); ++i) {
    if (threadStreams[i] != nullptr && default_streams_[i] != threadStreams[i])
      delete threadStreams[i];
    delete default_streams_[i];
  }
  default_streams_.resize(devices.size());
  threadStreams.resize(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    Stream* stream = new Stream(devices[i]);
    default_streams_[i] = stream;
    threadStreams[i] = stream;
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
  devices.push_back(d);
#else
  NSArray<id<MTLDevice> >* devs = MTLCopyAllDevices();
  for (size_t i = 0; i < devs.count; ++i) {
    id<MTLDevice> d = [devs objectAtIndex:i];
    devices.push_back(d);
    DLOG(INFO) << "Intializing Metal device " << i << ", name=" << [d.name UTF8String];
    warp_size.push_back(GetWarpSize(d));
  }
#endif
  ReinitializeStreams();
}

void MetalWorkspace::SetDevice(Device dev) {
  MetalThreadEntry::ThreadLocal()->device.device_id = dev.device_id;
}

void* MetalWorkspace::AllocDataSpace(Device device, size_t nbytes, size_t alignment,
                                     DLDataType type_hint) {
  id<MTLBuffer> buf;
  AUTORELEASEPOOL {
    this->Init();
    id<MTLDevice> dev = GetDevice(device);
    // GPU memory only
    MTLResourceOptions storage_mode = MTLResourceStorageModePrivate;
    /*
    #if TARGET_OS_IPHONE
    storage_mode = MTLResourceStorageModeShared;
    #else
    storage_mode = MTLResourceStorageModeManaged;
    #endif
    */
    buf = [dev newBufferWithLength:nbytes options:storage_mode];
    ICHECK(buf != nil);
  };
  return (void*)(buf);
}

void MetalWorkspace::FreeDataSpace(Device dev, void* ptr) {
  AUTORELEASEPOOL {
    // need to make sure buffer is not in use in command buffer
    // before set the purgeable state to empty
    // otherwise can cause issues sometimes
    this->StreamSync(dev, nullptr);
    // MTLBuffer PurgeableState should be set to empty before manual
    // release in order to prevent memory leak
    [(id<MTLBuffer>)ptr setPurgeableState:MTLPurgeableStateEmpty];
    // release the ptr.
    CFRelease(ptr);
  };
}

Stream* CastStreamOrGetCurrent(TVMStreamHandle stream, int device_id) {
  if (stream != nullptr) return static_cast<Stream*>(stream);
  ICHECK(MetalThreadEntry::ThreadLocal()->stream[device_id] != nullptr);
  return MetalThreadEntry::ThreadLocal()->stream[device_id];
}

void MetalWorkspace::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                    size_t to_offset, size_t size, Device dev_from, Device dev_to,
                                    DLDataType type_hint, TVMStreamHandle stream) {
  AUTORELEASEPOOL {
    this->Init();
    Device dev = dev_from;
    if (dev_from.device_type == kDLCPU) dev = dev_to;
    Stream* s = CastStreamOrGetCurrent(stream, dev.device_id);
    if (s->HasErrorHappened()) {
      LOG(FATAL) << "Error! Some problems on GPU happaned! Cannot copy data to current stream";
    }
    id<MTLCommandBuffer> cb = s->GetCommandBuffer();
    int from_dev_type = static_cast<int>(dev_from.device_type);
    int to_dev_type = static_cast<int>(dev_to.device_type);

    if (from_dev_type == kDLMetal && to_dev_type == kDLMetal) {
      ICHECK_EQ(dev_from.device_id, dev_to.device_id) << "Metal disallow cross device copy.";
      id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
      [encoder copyFromBuffer:(id<MTLBuffer>)(from)
                 sourceOffset:from_offset
                     toBuffer:(id<MTLBuffer>)(to)destinationOffset:to_offset
                         size:size];
      [encoder endEncoding];
      [cb commit];
    } else if (from_dev_type == kDLMetal && to_dev_type == kDLCPU) {
      // copy to a local buffer before get into global buffer.
      id<MTLBuffer> from_buf = (id<MTLBuffer>)(from);
      if (from_buf.storageMode != MTLStorageModeShared) {
        id<MTLBuffer> temp = MetalThreadEntry::ThreadLocal()->GetTempBuffer(dev_from, size);
        id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
        [encoder copyFromBuffer:from_buf
                   sourceOffset:from_offset
                       toBuffer:temp
              destinationOffset:0
                           size:size];
        [encoder endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(static_cast<char*>(to) + to_offset, static_cast<char*>([temp contents]), size);
      } else {
        memcpy(static_cast<char*>(to) + to_offset,
               static_cast<char*>([from_buf contents]) + from_offset, size);
      }
    } else if (from_dev_type == kDLCPU && to_dev_type == kDLMetal) {
      id<MTLBuffer> to_buf = (id<MTLBuffer>)(to);
      if (to_buf.storageMode != MTLStorageModeShared) {
        id<MTLBuffer> temp = MetalThreadEntry::ThreadLocal()->GetTempBuffer(dev_to, size);
        memcpy([temp contents], static_cast<const char*>(from) + from_offset, size);
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
               static_cast<const char*>(from) + from_offset, size);
      }
    } else {
      LOG(FATAL) << "Expect copy from/to Metal or between Metal"
                 << ", from=" << from_dev_type << ", to=" << to_dev_type;
    }
  };
}

TVMStreamHandle MetalWorkspace::CreateStream(Device dev) {
  ICHECK_LT(dev.device_id, devices.size()) << "Invalid device id " << dev.device_id;
  Stream* stream = new Stream(devices[dev.device_id]);
  return static_cast<TVMStreamHandle>(stream);
}

void MetalWorkspace::FreeStream(Device dev, TVMStreamHandle stream) {
  ICHECK(stream != nullptr);
  ICHECK_LT(dev.device_id, devices.size()) << "Invalid device id " << dev.device_id;
  Stream* s = static_cast<Stream*>(stream);
  if (MetalThreadEntry::ThreadLocal()->stream[dev.device_id] == s)
    MetalThreadEntry::ThreadLocal()->stream[dev.device_id] = nullptr;
  delete s;
}

void MetalWorkspace::StreamSync(Device dev, TVMStreamHandle stream) {
  AUTORELEASEPOOL {
    Stream* s = CastStreamOrGetCurrent(stream, dev.device_id);
    // commit an empty command buffer and wait until it completes.
    id<MTLCommandBuffer> cb = s->GetCommandBuffer();
    [cb commit];
    [cb waitUntilCompleted];
    if (s->HasErrorHappened()) {
      LOG(FATAL) << "Error! Some problems on GPU happaned!";
    }
  };
}

void MetalWorkspace::SetStream(Device dev, TVMStreamHandle stream) {
  ICHECK_LT(dev.device_id, devices.size()) << "Invalid device id " << dev.device_id;
  ICHECK(stream != nullptr);
  MetalThreadEntry::ThreadLocal()->stream[dev.device_id] = static_cast<Stream*>(stream);
}

void* MetalWorkspace::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return MetalThreadEntry::ThreadLocal()->pool.AllocWorkspace(dev, size);
}

void MetalWorkspace::FreeWorkspace(Device dev, void* data) {
  MetalThreadEntry::ThreadLocal()->pool.FreeWorkspace(dev, data);
}

MetalThreadEntry::~MetalThreadEntry() {
  for (auto x : temp_buffer_) {
    if (x != nil) {
      [(id<MTLBuffer>)x setPurgeableState:MTLPurgeableStateEmpty];
      [x release];
    }
  }
}

id<MTLBuffer> MetalThreadEntry::GetTempBuffer(Device dev, size_t size) {
  if (temp_buffer_.size() <= static_cast<size_t>(dev.device_id)) {
    temp_buffer_.resize(dev.device_id + 1, nil);
  }
  if (temp_buffer_[dev.device_id] == nil || temp_buffer_[dev.device_id].length < size) {
    id<MTLDevice> mtl_dev = MetalWorkspace::Global()->GetDevice(dev);
    if (temp_buffer_[dev.device_id] != nil) {
      // need to make sure buffer is not in use in command buffer
      // before set the purgeable state to empty
      // otherwise can cause issues sometimes
      MetalWorkspace::Global()->StreamSync(dev, nullptr);
      [temp_buffer_[dev.device_id] setPurgeableState:MTLPurgeableStateEmpty];
      [temp_buffer_[dev.device_id] release];
    }
    temp_buffer_[dev.device_id] = [mtl_dev newBufferWithLength:size options:MTLStorageModeShared];
  }
  return temp_buffer_[dev.device_id];
}

typedef dmlc::ThreadLocalStore<MetalThreadEntry> MetalThreadStore;

MetalThreadEntry* MetalThreadEntry::ThreadLocal() { return MetalThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.metal").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = MetalWorkspace::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_GLOBAL("metal.ResetGlobalState").set_body_typed([]() {
  MetalWorkspace::Global()->ReinitializeStreams();
});

}  // namespace metal
}  // namespace runtime
}  // namespace tvm
