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
  MicroDeviceAPI() {
    session_ = MicroSession::Global();
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
    void* alloc_ptr = session_->AllocateInSection(kHeap, nbytes);
    return alloc_ptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    session_->FreeInSection(kHeap, ptr);
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

    if (type_from_to == std::make_tuple(micro_devtype, micro_devtype)) {
      CHECK(ctx_from.device_id == ctx_to.device_id)
        << "can only copy between the same micro device";
      std::string buffer;
      const std::shared_ptr<LowLevelDevice>& from_lld = session_->low_level_device();
      const std::shared_ptr<LowLevelDevice>& to_lld = session_->low_level_device();
      from_lld->Read(
          const_cast<uint8_t*>(static_cast<const uint8_t*>(from)) + from_offset,
          const_cast<char*>(&buffer[0]), size);
      to_lld->Write(
          const_cast<uint8_t*>(static_cast<const uint8_t*>(to)) + to_offset,
          const_cast<char*>(&buffer[0]), size);
    } else if (type_from_to == std::make_tuple(micro_devtype, kDLCPU)) {
      const std::shared_ptr<LowLevelDevice>& from_lld = session_->low_level_device();
      from_lld->Read(
          const_cast<uint8_t*>(static_cast<const uint8_t*>(from)) + from_offset,
          const_cast<uint8_t*>(static_cast<const uint8_t*>(to)), size);

    } else if (type_from_to == std::make_tuple(micro_devtype, kDLCPU)) {
      const std::shared_ptr<LowLevelDevice>& to_lld = session_->low_level_device();
      to_lld->Write(
          const_cast<uint8_t*>(static_cast<const uint8_t*>(to)) + to_offset,
          const_cast<uint8_t*>(static_cast<const uint8_t*>(from)) + from_offset,
          size);

    } else {
      LOG(FATAL) << "Expect copy from/to micro_dev or between micro_dev\n";
    }
  }

  // TODO(): ignore this?
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }

  // TODO: what about ctx?
  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final {
    void* alloc_ptr = session_->AllocateInSection(kWorkspace, size);
    return alloc_ptr;
  }

  // TODO: what about ctx?
  void FreeWorkspace(TVMContext ctx, void* data) final {
    session_->FreeInSection(kWorkspace, data);
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
  MicroSession* session_;
};

// register device that can be obtained from Python frontend
TVM_REGISTER_GLOBAL("device_api.micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = MicroDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });
}  // namespace runtime
}  // namespace tvm
