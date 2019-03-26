/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_device_api.cc
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include "../workspace_pool.h"

namespace tvm {
namespace runtime {
/*!
 * \brief device API for uTVM micro devices
 */
class MicroDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {}

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final {
    return nullptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
  }  

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMType type_hint,
                      TVMStreamHandle stream) final {
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final {
    return nullptr;
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
  }

  /*!
   * \brief obtain a global singleton of MicroDeviceAPI
   * \return global shared pointer to MicroDeviceAPI
   */
  static const std::shared_ptr<MicroDeviceAPI>& Global() {
    static std::shared_ptr<MicroDeviceAPI> inst =
        std::make_shared<MicroDeviceAPI>();
    return inst;
  }
};

// register device that can be obtained from Python frontend
TVM_REGISTER_GLOBAL("device_api.micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = MicroDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });
} // namespace runtime
} // namespace tvm
