/*!
 *  Copyright (c) 2017 by Contributors
 * \file metal_device_api.mm
 */
#include "./metal_common.h"

#if TVM_METAL_RUNTIME
#include <tvm/runtime/registry.h>
#include <dmlc/thread_local.h>

namespace tvm {
namespace runtime {
namespace metal {

MetalWorkspace* MetalWorkspace::Global() {
  static MetalWorkspace inst;
  return &inst;
}

void MetalWorkspace::GetAttr(
    int dev_id, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(dev_id);
  if (kind == kExist) {
    *rv = int(index< devices.size());
    return;
  }
  CHECK_LT(index, devices.size())
      << "Invalid device id " << index;
  switch (kind) {
    case kMaxThreadsPerBlock: {
      *rv = static_cast<int>(
          [devices[dev_id] maxThreadsPerThreadgroup].width);
      break;
    }
    case kWarpSize: {
      // Set warp size to be 1 for safty reason.
      *rv = 1;
      break;
    }
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
  CHECK(lib != nil) << error_msg;
  id<MTLFunction> f =
      [lib
        newFunctionWithName:
          [NSString stringWithUTF8String:"CopyKernel"]];
  CHECK(f!= nil);
  id<MTLComputePipelineState> state =
      [dev
        newComputePipelineStateWithFunction:f
        error:&error_msg];
  CHECK(state != nil) << error_msg;
  return state.threadExecutionWidth;
}

void MetalWorkspace::Init() {
  if (devices.size() != 0) return;
  std::lock_guard<std::mutex>(this->mutex);
  if (devices.size() != 0) return;
  NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
  for (size_t i = 0; i < devs.count; ++i) {
    id<MTLDevice> d = [devs objectAtIndex:i];
    devices.push_back(d);
    queues.push_back([d newCommandQueue]);
    LOG(INFO) << "Intializing Metal device " << i
              <<  ", name=" << d.name;
    warp_size.push_back(GetWarpSize(d));
  }
}

void MetalWorkspace::SetDevice(int dev_id) {
  MetalThreadEntry::ThreadLocal()->context.device_id = dev_id;
}

void* MetalWorkspace::AllocDataSpace(
    TVMContext ctx, size_t size, size_t alignment) {
  this->Init();
  id<MTLDevice> dev = GetDevice(ctx);
  // allocate buffer in GPU only mode.
  id<MTLBuffer> buf = [
      dev newBufferWithLength:size
          options:MTLResourceStorageModePrivate];
  // retain ARC to keep it alive before release.
  return (__bridge_retained void*)(buf);
}

void MetalWorkspace::FreeDataSpace(TVMContext ctx, void* ptr) {
  // release the ptr.
  CFBridgingRelease(ptr);
}

void MetalWorkspace::CopyDataFromTo(const void* from,
                                    size_t from_offset,
                                    void* to,
                                    size_t to_offset,
                                    size_t size,
                                    TVMContext ctx_from,
                                    TVMContext ctx_to,
                                    TVMStreamHandle stream) {
  this->Init();
  CHECK(stream == nullptr);
  TVMContext ctx = ctx_from;
  if (ctx_from.device_type == kCPU) ctx = ctx_to;
  id<MTLCommandQueue> queue = GetCommandQueue(ctx);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
  int from_dev_type = static_cast<int>(ctx_from.device_type);
  int to_dev_type = static_cast<int>(ctx_to.device_type);

  if (from_dev_type == kMetal && to_dev_type == kMetal) {
    CHECK_EQ(ctx_from.device_id, ctx_to.device_id)
        << "Metal disallow cross device copy.";
    [encoder copyFromBuffer:(__bridge id<MTLBuffer>)(from)
             sourceOffset:from_offset
             toBuffer:(__bridge id<MTLBuffer>)(to)
             destinationOffset:to_offset
             size:size];
    [encoder endEncoding];
    [cb commit];
  } else if (from_dev_type == kMetal && to_dev_type == kCPU) {
    // copy to a local buffer before get into global buffer.
    id<MTLBuffer> from_buf = (__bridge id<MTLBuffer>)(from);
    if (from_buf.storageMode != MTLStorageModeShared) {
      id<MTLBuffer> temp = MetalThreadEntry::ThreadLocal()
          ->GetTempBuffer(ctx_from, size);
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
  } else if (from_dev_type == kCPU && to_dev_type == kMetal) {
    id<MTLBuffer> to_buf = (__bridge id<MTLBuffer>)(to);
    if (to_buf.storageMode == MTLStorageModeShared) {
      id<MTLBuffer> temp = MetalThreadEntry::ThreadLocal()
          ->GetTempBuffer(ctx_to, size);
      memcpy([temp contents],
              static_cast<const char*>(from) + from_offset,
              size);
      [encoder copyFromBuffer:temp
               sourceOffset:0
               toBuffer:to_buf
               destinationOffset:to_offset
               size:size];
      [encoder endEncoding];
      [cb commit];
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

id<MTLBuffer> MetalThreadEntry::GetTempBuffer(TVMContext ctx, size_t size) {
  if (temp_buffer_.size() <= static_cast<size_t>(ctx.device_id)) {
    temp_buffer_.resize(ctx.device_id + 1, nil);
  }
  if (temp_buffer_[ctx.device_id] == nil ||
      temp_buffer_[ctx.device_id].length < size) {
    id<MTLDevice> dev = MetalWorkspace::Global()->GetDevice(ctx);
    temp_buffer_[ctx.device_id] = [
        dev newBufferWithLength:size
            options:MTLStorageModeShared];
  }
  return temp_buffer_[ctx.device_id];
}

typedef dmlc::ThreadLocalStore<MetalThreadEntry> MetalThreadStore;

MetalThreadEntry* MetalThreadEntry::ThreadLocal() {
  return MetalThreadStore::Get();
}

TVM_REGISTER_GLOBAL("device_api.metal")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = MetalWorkspace::Global();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace metal
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_METAL_RUNTIME
