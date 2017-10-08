/*!
 *  Copyright (c) 2016 by Contributors
 * \file device_api.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RUNTIME_DEVICE_API_H_
#define TVM_RUNTIME_DEVICE_API_H_

#include <string>
#include "./packed_func.h"
#include "./c_runtime_api.h"

namespace tvm {
namespace runtime {
/*!
 * \brief the query type into GetAttr
 */
enum DeviceAttrKind : int {
  kExist = 0,
  kMaxThreadsPerBlock = 1,
  kWarpSize = 2,
  kComputeVersion = 3
};

/*! \brief Number of bytes each allocation must align to */
constexpr int kAllocAlignment = 64;

/*! \brief Number of bytes each allocation must align to in temporary allocation */
constexpr int kTempAllocaAlignment = 64;

/*! \brief Maximum size that can be allocated on stack */
constexpr int kMaxStackAlloca = 1024;

/*!
 * \brief TVM Runtime Device API, abstracts the device
 *  specific interface for memory management.
 */
class DeviceAPI {
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
   * \param size The size of the memory
   * \param alignment The alignment of the memory.
   * \return The allocated device pointer
   */
  virtual void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment) = 0;
  /*!
   * \brief Free a data space on device.
   * \param ctx The device context to perform operation.
   * \param ptr The data space.
   */
  virtual void FreeDataSpace(TVMContext ctx, void* ptr) = 0;
  /*!
   * \brief copy data from one place to another
   * \param from The source array.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param size The size of the memory
   * \param ctx_from The source context
   * \param ctx_to The target context
   * \param stream Optional stream object.
   */
  virtual void CopyDataFromTo(const void* from,
                              size_t from_offset,
                              void* to,
                              size_t to_offset,
                              size_t size,
                              TVMContext ctx_from,
                              TVMContext ctx_to,
                              TVMStreamHandle stream) = 0;
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
   * \param size The size to be allocated.
   */
  TVM_DLL virtual void* AllocWorkspace(TVMContext ctx, size_t size);
  /*!
   * \brief Free temporal workspace in backend execution.
   *
   * \param ctx The context of allocation.
   * \param ptr The pointer to be freed.
   */
  TVM_DLL virtual void FreeWorkspace(TVMContext ctx, void* ptr);
  /*!
   * \brief Get device API base don context.
   * \param ctx The context
   * \param allow_missing Whether allow missing
   * \return The corresponding device API.
   */
  TVM_DLL static DeviceAPI* Get(TVMContext ctx, bool allow_missing = false);
};

/*! \brief The device type bigger than this is RPC device */
constexpr int kRPCSessMask = 128;
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DEVICE_API_H_
