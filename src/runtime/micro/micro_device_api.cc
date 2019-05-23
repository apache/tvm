/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_device_api.cc
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include "../workspace_pool.h"
#include "micro_session.h"

namespace tvm {
namespace runtime {
/*!
 * \brief device API for uTVM micro devices
 */
class MicroDeviceAPI final : public DeviceAPI {
 public:
  /*! \brief constructor */
  MicroDeviceAPI()
    : session_(MicroSession::Global()) {
  }

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
    return session_->AllocateInSection(kHeap, nbytes).cast_to<void*>();
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    session_->FreeInSection(kHeap, DevBaseOffset(reinterpret_cast<std::uintptr_t>(ptr)));
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
    constexpr int micro_devtype = kDLMicroDev;
    std::tuple<int, int> type_from_to(ctx_from.device_type, ctx_to.device_type);
    DevBaseOffset from_base_offset =
        DevBaseOffset(reinterpret_cast<std::uintptr_t>(const_cast<void*>(from)) + from_offset);
    DevBaseOffset to_base_offset =
        DevBaseOffset(reinterpret_cast<std::uintptr_t>(const_cast<void*>(to)) + to_offset);
    const std::shared_ptr<LowLevelDevice>& lld = session_->low_level_device();

    if (type_from_to == std::make_tuple(micro_devtype, micro_devtype)) {
      // Copying from the device to the device.
      CHECK(ctx_from.device_id == ctx_to.device_id)
        << "can only copy between the same micro device";
      std::vector<uint8_t> buffer(size);
      lld->Read(from_base_offset, reinterpret_cast<void*>(buffer.data()), size);
      lld->Write(to_base_offset, reinterpret_cast<void*>(buffer.data()), size);
    } else if (type_from_to == std::make_tuple(micro_devtype, kDLCPU)) {
      // Reading from the device.
      const std::shared_ptr<LowLevelDevice>& from_lld = session_->low_level_device();
      lld->Read(from_base_offset, to_base_offset.cast_to<void*>(), size);
    } else if (type_from_to == std::make_tuple(kDLCPU, micro_devtype)) {
      // Writing to the device.
      const std::shared_ptr<LowLevelDevice>& to_lld = session_->low_level_device();
      lld->Write(to_base_offset, from_base_offset.cast_to<void*>(), size);

    } else {
      LOG(FATAL) << "Expect copy from/to micro_dev or between micro_dev\n";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final {
    return session_->AllocateInSection(kWorkspace, size).cast_to<void*>();
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    session_->FreeInSection(kWorkspace, DevBaseOffset(reinterpret_cast<std::uintptr_t>(data)));
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

 private:
  /*! \brief pointer to global session */
  std::shared_ptr<MicroSession> session_;
};

// register device that can be obtained from Python frontend
TVM_REGISTER_GLOBAL("device_api.micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = MicroDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
    });
}  // namespace runtime
}  // namespace tvm
