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
 * \file tvm/runtime/device_api.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RUNTIME_DEVICE_API_H_
#define TVM_RUNTIME_DEVICE_API_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>

#include <string>

namespace tvm {
namespace runtime {
/*!
 * \brief the query type into GetAttr
 */
enum DeviceAttrKind : int {
  kExist = 0,
  kMaxThreadsPerBlock = 1,
  kWarpSize = 2,
  kMaxSharedMemoryPerBlock = 3,
  kComputeVersion = 4,
  kDeviceName = 5,
  kMaxClockRate = 6,
  kMultiProcessorCount = 7,
  kMaxThreadDimensions = 8,
  kMaxRegistersPerBlock = 9,
  kGcnArch = 10,
  kApiVersion = 11
};

/*! \brief Number of bytes each allocation must align to */
constexpr int kAllocAlignment = 128;

/*! \brief Number of bytes each allocation must align to in temporary allocation */
constexpr int kTempAllocaAlignment = 128;

/*! \brief Maximum size that can be allocated on stack */
constexpr int kMaxStackAlloca = 1024;

/*!
 *  \brief TVM Runtime Device API, abstracts the device
 *  specific interface for memory management.
 */
class TVM_DLL DeviceAPI {
 public:
  /*! \brief virtual destructor */
  virtual ~DeviceAPI() {}
  /*!
   * \brief Set the environment device id to ctx
   * \param ctx The context to be set.
   */
  virtual void SetDevice(TVMContext ctx) = 0;
  /*!
   * \brief Get attribute of specified device.
   * \param ctx The device context
   * \param kind The result kind
   * \param rv The return value.
   * \sa DeviceAttrKind
   */
  virtual void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) = 0;
  /*!
   * \brief Allocate a data space on device.
   * \param ctx The device context to perform operation.
   * \param nbytes The number of bytes in memory.
   * \param alignment The alignment of the memory.
   * \param type_hint The type of elements. Only needed by certain backends such
   * as OpenGL, as nbytes & alignment are sufficient for most backends.
   * \return The allocated device pointer.
   */
  virtual void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
                               DLDataType type_hint) = 0;
  /*!
   * \brief Allocate a data space on device with memory scope support.
   * \param ctx The device context to perform operation.
   * \param ndim The number of dimension of allocated tensor.
   * \param shape The shape of allocated tensor.
   * \param dtype The type of elements.
   * \param mem_scope The memory scope of allocated tensor.
   * \return The allocated device pointer.
   */
  virtual void* AllocDataSpace(TVMContext ctx, int ndim, const int64_t* shape, DLDataType dtype,
                               Optional<String> mem_scope = NullOpt);
  /*!
   * \brief Free a data space on device.
   * \param ctx The device context to perform operation.
   * \param ptr The data space.
   */
  virtual void FreeDataSpace(TVMContext ctx, void* ptr) = 0;
  /*!
   * \brief copy data from one place to another
   * \note This API is designed to support special memory with shape dependent layout.
   *       We pass in DLTensor* with shape information to support these cases.
   * \param from The source array.
   * \param to The target array.
   * \param stream Optional stream object.
   */
  virtual void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream);
  /*!
   * \brief Create a new stream of execution.
   *
   * \param ctx The context of allocation.
   */
  virtual TVMStreamHandle CreateStream(TVMContext ctx);

  /*!
   * \brief Free a stream of execution
   *
   * \param ctx The context of the stream
   * \param stream The pointer to be freed.
   */
  virtual void FreeStream(TVMContext ctx, TVMStreamHandle stream);

  /*!
   * \brief Synchronize the stream
   * \param ctx The context to perform operation.
   * \param stream The stream to be sync.
   */
  virtual void StreamSync(TVMContext ctx, TVMStreamHandle stream) = 0;
  /*!
   * \brief Set the stream
   * \param ctx The context to set stream.
   * \param stream The stream to be set.
   */
  virtual void SetStream(TVMContext ctx, TVMStreamHandle stream) {}
  /*!
   * \brief Synchronize 2 streams of execution.
   *
   * An event is created in event_src stream that the second then
   * stream waits on.  Neither event_src or event_dst need to be of
   * the same device ID as the context, but they must be of the same
   * device type.
   *
   * \param ctx The context of the streams.
   * \param event_src The source stream to synchronize.
   * \param event_dst The destination stream to synchronize.
   */
  virtual void SyncStreamFromTo(TVMContext ctx, TVMStreamHandle event_src,
                                TVMStreamHandle event_dst);
  /*!
   * \brief Allocate temporal workspace for backend execution.
   *
   *  \note We have the following assumption about backend temporal
   *   workspace allocation, and backend will optimize for such assumption:
   *
   *  - Only a few allocation will happen, and space will be released after use.
   *  - The release order is usually in reverse order of allocate (stack style).
   *  - Repeative pattern of same allocations over different runs.
   *  - Workspace should not overlap between different threads(i.e. be threadlocal)
   *
   * \param ctx The context of allocation.
   * \param nbytes The size to be allocated.
   * \param type_hint The type of elements. Only needed by certain backends such
   * as OpenGL, as nbytes is sufficient for most backends.
   */
  virtual void* AllocWorkspace(TVMContext ctx, size_t nbytes, DLDataType type_hint = {});
  /*!
   * \brief Free temporal workspace in backend execution.
   *
   * \param ctx The context of allocation.
   * \param ptr The pointer to be freed.
   */
  virtual void FreeWorkspace(TVMContext ctx, void* ptr);

  /*!
   * \brief Get device API based on context.
   * \param ctx The context
   * \param allow_missing Whether allow missing
   * \return The corresponding device API.
   */
  static DeviceAPI* Get(TVMContext ctx, bool allow_missing = false);

  /*!
   * \brief Whether a certian device type requires set device context
   *        before launching the kernel function.
   * \param device_type The device type.
   */
  static bool NeedSetDeviceContext(int device_type) {
    return device_type != kDLCPU && device_type != kDLMicroDev;
  }

 protected:
  /*!
   * \brief copy data from one place to another
   * \param from The source array.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param num_bytes The size of the memory in bytes
   * \param ctx_from The source context
   * \param ctx_to The target context
   * \param type_hint The type of elements, only neded by certain backends.
   *                  can be useful for cross device endian converison.
   * \param stream Optional stream object.
   */
  virtual void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                              size_t num_bytes, TVMContext ctx_from, TVMContext ctx_to,
                              DLDataType type_hint, TVMStreamHandle stream);
};

/*! \brief The device type bigger than this is RPC device */
constexpr int kRPCSessMask = 128;

/*!
 * \brief The name of Device API factory.
 * \param type The device type.
 * \return the device name.
 */
inline const char* DeviceName(int type) {
  switch (type) {
    case kDLCPU:
      return "cpu";
    case kDLGPU:
      return "gpu";
    case kDLCPUPinned:
      return "cpu_pinned";
    case kDLOpenCL:
      return "opencl";
    case kDLSDAccel:
      return "sdaccel";
    case kDLAOCL:
      return "aocl";
    case kDLVulkan:
      return "vulkan";
    case kDLMetal:
      return "metal";
    case kDLVPI:
      return "vpi";
    case kDLROCM:
      return "rocm";
    case kDLExtDev:
      return "ext_dev";
    case kDLWebGPU:
      return "webgpu";
    case kDLMicroDev:
      return "micro_dev";
    case kDLHexagon:
      return "hexagon";
    default:
      LOG(FATAL) << "unknown type =" << type;
      return "Unknown";
  }
}

/*!
 * \brief Return true if a TVMContext is owned by an RPC session.
 */
inline bool IsRPCSessionContext(TVMContext ctx) { return (ctx.device_type / kRPCSessMask) > 0; }

/*!
 * \brief Return the RPCSessTable index of the RPC Session that owns this context.
 * \return the table index.
 */
inline int GetRPCSessionIndex(TVMContext ctx) {
  ICHECK(IsRPCSessionContext(ctx)) << "GetRPCSessionIndex: ctx has no RPC session";
  return ctx.device_type / kRPCSessMask - 1;
}

/*!
 * \brief Remove the RPC session mask from a TVMContext.
 * RPC clients typically do this when encoding a TVMContext for transmission to an RPC remote.
 * On the wire, RPCContext are expected to be valid on the server without interpretation.
 * \param ctx A TVMContext with non-zero RPC Session mask, valid on the RPC client.
 * \return A TVMContext without any RPC Session mask, valid on the RPC server.
 */
inline TVMContext RemoveRPCSessionMask(TVMContext ctx) {
  ctx.device_type = static_cast<DLDeviceType>(ctx.device_type % kRPCSessMask);
  return ctx;
}

inline std::ostream& operator<<(std::ostream& os, DLContext ctx);

/*!
 * \brief Add a RPC session mask to a TVMContext.
 * RPC clients typically do this when decoding a TVMContext received from a RPC remote.
 * \param ctx A TVMContext without any RPC Session mask, valid on the RPC server.
 * \param session_table_index Numeric index of the RPC session in the session table.
 * \return A TVMContext with RPC session mask added, valid on the RPC client.
 */
inline TVMContext AddRPCSessionMask(TVMContext ctx, int session_table_index) {
  CHECK(!IsRPCSessionContext(ctx))
      << "AddRPCSessionMask: ctx already non-zero RPCSessionIndex: " << ctx;
  ctx.device_type =
      static_cast<DLDeviceType>(ctx.device_type | (kRPCSessMask * (session_table_index + 1)));
  return ctx;
}

inline std::ostream& operator<<(std::ostream& os, DLContext ctx) {  // NOLINT(*)
  if (IsRPCSessionContext(ctx)) {
    os << "remote[" << GetRPCSessionIndex(ctx) << "]-";
    ctx = RemoveRPCSessionMask(ctx);
  }
  os << runtime::DeviceName(static_cast<int>(ctx.device_type)) << "(" << ctx.device_id << ")";
  return os;
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DEVICE_API_H_
