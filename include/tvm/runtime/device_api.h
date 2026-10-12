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

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/device.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/base.h>

#include <string>
/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void* TVMStreamHandle;

namespace tvm {

// alias DLDevice
using Device = DLDevice;

namespace runtime {

/*! \brief Extension device types in TVM
 *
 * Additional enumerators to supplement those provided by
 * DLPack's `DLDeviceType` enumeration.
 *
 * MAINTAINERS NOTE #1: We need to ensure that the two devices
 * are identified by the same integer.
 * Currently this requires manual verification.
 * Discussed here: https://github.com/dmlc/dlpack/issues/111
 * As of DLPack v0.7, the highest-valued enumerator in
 * `DLDeviceType` is kDLHexagon = 16.
 *
 * MAINTAINERS NOTE #2: As of DLPack v0.7, the definition for
 * `DLDeviceType` specifies an underlying storage type of
 * `int32_t`.  That guarantees a variable of type
 * `DLDeviceType` is capable of holding any integers provided
 * by *either* of these enumerations.
 *
 * However, the `int32_t` specification only applies when the
 * header file is compiled as C++, and this header file is also
 * meant to work as C code.  So the unspecified storage type
 * could be a latent bug when compiled as C.
 */
#ifdef __cplusplus
typedef enum : int32_t {
#else
typedef enum {
#endif
  // To help avoid accidental conflicts between `DLDeviceType`
  // and this enumeration, start numbering the new enumerators
  // a little higher than (currently) seems necessary.
  TVMDeviceExtType_End = 36,  // sentinel value
} TVMDeviceExtType;

#ifdef TVM_KALLOC_ALIGNMENT
/*! \brief Number of bytes each allocation must align to */
constexpr int kAllocAlignment = TVM_KALLOC_ALIGNMENT;

/*! \brief Number of bytes each allocation must align to in temporary allocation */
constexpr int kTempAllocaAlignment = TVM_KALLOC_ALIGNMENT;
#else
/*! \brief Number of bytes each allocation must align to */
constexpr int kAllocAlignment = 64;

/*! \brief Number of bytes each allocation must align to in temporary allocation */
constexpr int kTempAllocaAlignment = 64;
#endif  // TVM_KALLOC_ALIGNMENT

/*! \brief Maximum size that can be allocated on stack */
constexpr int kMaxStackAlloca = 1024;

/*! \brief Number of bytes each allocation must align to by default in the workspace buffer to
 * service intermediate tensors */
constexpr int kDefaultWorkspaceAlignment = 1;

/*!
 *  \brief TVM Runtime Device API, abstracts the device
 *  specific interface for memory management.
 */
class TVM_RUNTIME_DLL DeviceAPI {
 public:
  /*! \brief virtual destructor */
  virtual ~DeviceAPI() {}
  /*!
   * \brief Set the environment device id to device
   * \param dev The device to be set.
   */
  virtual void SetDevice(Device dev) = 0;
  /*!
   * \brief Get the currently selected runtime device index on the calling host thread.
   *
   * The device type is determined by this DeviceAPI instance. The result is a
   * runtime device index (DLDevice::device_id), not a device type or a target
   * configuration index. Implementations must query the current selection without
   * changing it. The default implementation throws NotImplementedError; support
   * for SetDevice does not imply support for this query. RPC devices are excluded.
   *
   * \return The selected device index, on success.
   */
  virtual int GetDeviceIndex();
  /*!
   * \brief Query an attribute of a device.
   * \param dev The device to query.
   * \param name Attribute name. "exist" returns a boolean; unsupported optional
   * attributes return None. Errors querying a supported attribute are propagated.
   * \return The attribute value.
   */
  virtual ffi::Any GetAttr(DLDevice dev, const ffi::String& name) = 0;

  /*!
   * \brief Create an owning, compact tensor with uninitialized contents.
   * \param shape Logical shape, copied into the returned tensor.
   * \param dtype Element datatype.
   * \param device Device on which to allocate storage.
   * \param mem_scope Optional backend memory scope.
   * \return A tensor whose deleter owns the backend allocation and its resources.
   */
  virtual ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                            ffi::Optional<ffi::String> mem_scope = std::nullopt) = 0;

  /*!
   * \brief copy data from one place to another
   * \note This API is designed to support special memory with shape dependent layout.
   *       We pass in DLTensor* with shape information to support these cases.
   * \param from The source array.
   * \param to The target array.
   * \param stream Optional stream object.
   * \note The copy may happen asynchronously if it involves a GPU context.
   *       Call StreamSync to ensure the copy completes from host's pov.
   */
  virtual void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream);
  /*!
   * \brief Create a new stream of execution.
   *
   * \param dev The device of allocation.
   */
  virtual TVMStreamHandle CreateStream(Device dev);

  /*!
   * \brief Free a stream of execution
   *
   * \param dev The device of the stream
   * \param stream The pointer to be freed.
   */
  virtual void FreeStream(Device dev, TVMStreamHandle stream);

  /*!
   * \brief Synchronize the stream
   * \param dev The device to perform operation.
   * \param stream The stream to be sync.
   */
  virtual void StreamSync(Device dev, TVMStreamHandle stream) = 0;
  /*!
   * \brief Set the stream
   * \param dev The device to set stream.
   * \param stream The stream to be set.
   */
  virtual void SetStream(Device dev, TVMStreamHandle stream);
  /*!
   * \brief Get the current stream
   * \param dev The device to get stream.
   * \return The current stream of the device.
   */
  virtual TVMStreamHandle GetCurrentStream(Device dev);
  /*!
   * \brief Synchronize 2 streams of execution.
   *
   * An event is created in event_src stream that the second then
   * stream waits on.  Neither event_src or event_dst need to be of
   * the same device ID as the device, but they must be of the same
   * device type.
   *
   * \param dev The device of the streams.
   * \param event_src The source stream to synchronize.
   * \param event_dst The destination stream to synchronize.
   */
  virtual void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst);
  /*!
   * \brief Get device API based on device.
   * \param dev The device
   * \param allow_missing Whether allow missing
   * \return The corresponding device API.
   */
  static DeviceAPI* Get(Device dev, bool allow_missing = false);

  /*!
   * \brief Whether a certian device type requires set device device
   *        before launching the kernel function.
   * \param device_type The device type.
   */
  static bool NeedSetDevice(int device_type) { return device_type != kDLCPU; }

  /*!
   * \brief Whether pointer arithmetics on a device owned pointer may be performed on the host.
   */
  virtual bool SupportsDevicePointerArithmeticsOnHost() { return false; }

 protected:
  /*!
   * \brief copy data from one place to another
   * \param from The source array.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param num_bytes The size of the memory in bytes
   * \param dev_from The source device
   * \param dev_to The target device
   * \param type_hint The type of elements, only neded by certain backends.
   *                  can be useful for cross device endian converison.
   * \param stream Optional stream object.
   */
  virtual void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                              size_t num_bytes, Device dev_from, Device dev_to,
                              DLDataType type_hint, TVMStreamHandle stream);
};

/*!
 * \brief The name of DLDeviceType.
 * \param type The device type.
 * \return the device name.
 */
inline const char* DLDeviceType2Str(int type) {
  switch (type) {
    case kDLCPU:
      return "cpu";
    case kDLCUDA:
      return "cuda";
    case kDLCUDAHost:
      return "cuda_host";
    case kDLCUDAManaged:
      return "cuda_managed";
    case kDLOpenCL:
      return "opencl";
    case kDLVulkan:
      return "vulkan";
    case kDLMetal:
      return "metal";
    case kDLVPI:
      return "vpi";
    case kDLROCM:
      return "rocm";
    case kDLROCMHost:
      return "rocm_host";
    case kDLExtDev:
      return "ext_dev";
    case kDLOneAPI:
      return "oneapi";
    case kDLWebGPU:
      return "webgpu";
    case kDLHexagon:
      return "hexagon";
    case kDLTrn:
      return "trn";
    default:
      TVM_FFI_THROW(InternalError) << "unknown type = " << type;
  }
  throw;
}

inline std::ostream& operator<<(std::ostream& os, DLDevice dev) {  // NOLINT(*)
  os << DLDeviceType2Str(static_cast<int>(dev.device_type)) << ":" << dev.device_id;
  return os;
}

/*!
 * \brief Check if runtime module is enabled for target.
 * \param target The target module name.
 * \return Whether runtime is enabled.
 */
TVM_RUNTIME_DLL bool RuntimeEnabled(const ffi::String& target);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DEVICE_API_H_
