/*!
 *  Copyright (c) 2016 by Contributors
 * \file device_api.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RUNTIME_DEVICE_API_H_
#define TVM_RUNTIME_DEVICE_API_H_

#include <string>
#include "./c_runtime_api.h"

namespace tvm {
namespace runtime {
/*!
 * \brief the query type into GetAttr
 */
enum DeviceAttrKind : int {
  kExist = 0,
  kMaxThreadsPerBlock = 1,
  kWarpSize = 2
};
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
   * \return The allocated device pointer
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
   * \tparam xpu The device mask.
   */
  virtual void FreeDataSpace(TVMContext ctx, void* ptr) = 0;
  /*!
   * \brief copy data from one place to another
   * \param dev The device to perform operation.
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
   * \brief Get device API base don context.
   * \param ctx The context
   * \param allow_missing Whether allow missing
   * \return The corresponding device API.
   */
  static DeviceAPI* Get(TVMContext ctx, bool allow_missing = false);
};

/*! \brief The device type bigger than this is RPC device */
constexpr int kRPCSessMask = 128;
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DEVICE_API_H_
