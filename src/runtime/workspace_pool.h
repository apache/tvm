/*!
 *  Copyright (c) 2017 by Contributors
 * \file workspace_pool.h
 * \brief Workspace pool utility.
 */
#ifndef TVM_RUNTIME_WORKSPACE_POOL_H_
#define TVM_RUNTIME_WORKSPACE_POOL_H_

#include <tvm/runtime/device_api.h>
#include <vector>

namespace tvm {
namespace runtime {
/*!
 * \brief A workspace pool to manage
 *
 *  \note We have the following assumption about backend temporal
 *   workspace allocation, and will optimize for such assumption,
 *   some of these assumptions can be enforced by the compiler.
 *
 *  - Only a few allocation will happen, and space will be released after use.
 *  - The release order is usually in reverse order of allocate
 *  - Repeative pattern of same allocations over different runs.
 */
class WorkspacePool {
 public:
  /*!
   * \brief Create pool with specific device type and device.
   * \param device_type The device type.
   * \param device The device API.
   */
  WorkspacePool(DLDeviceType device_type, std::shared_ptr<DeviceAPI> device);
  /*! \brief destructor */
  ~WorkspacePool();
  /*!
   * \brief Allocate temporal workspace.
   * \param ctx The context of allocation.
   * \param size The size to be allocated.
   */
  void* AllocWorkspace(TVMContext ctx, size_t size);
  /*!
   * \brief Free temporal workspace in backend execution.
   *
   * \param ctx The context of allocation.
   * \param ptr The pointer to be freed.
   */
  void FreeWorkspace(TVMContext ctx, void* ptr);

 private:
  class Pool;
  /*! \brief pool of device local array */
  std::vector<Pool*> array_;
  /*! \brief device type this pool support */
  DLDeviceType device_type_;
  /*! \brief The device API */
  std::shared_ptr<DeviceAPI> device_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_WORKSPACE_POOL_H_
