/*!
 *  Copyright (c) 2019 by Contributors
 * \file ffi_device_api.cc
 */
#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>

namespace tvm {
namespace runtime {

class FFIDeviceAPI final : public DeviceAPI {
 public:
  explicit FFIDeviceAPI(std::string device_name) : device_name_(device_name) {
    Registry::Register("device_api." + device_name_, false /* override */)
    .set_body([&](TVMArgs args, TVMRetValue* rv) { *rv = this; });
  }

  ~FFIDeviceAPI() {
    Registry::Remove("device_api." + device_name_);
  }

  void SetDevice(TVMContext ctx) final {
    (*GetFn("set_device"))(ctx);
  }

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    *rv = (*GetFn("get_attr"))(ctx, static_cast<int>(kind));
  }

  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final {
    return (*GetFn("alloc_data_space"))(ctx, nbytes, alignment, type_hint);
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    (*GetFn("free_data_space"))(ctx, ptr);
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
    (*GetFn("copy_data_from_to"))(
        const_cast<void*>(from), from_offset,
        to, to_offset, size,
        ctx_from, ctx_to,
        type_hint, static_cast<void*>(stream));
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    (*GetFn("stream_sync"))(ctx, reinterpret_cast<void*>(stream));
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final {
    return (*GetFn("alloc_workspace"))(ctx, size, type_hint);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    (*GetFn("free_workspace"))(ctx, data);
  }

 private:
  std::string device_name_;
  const PackedFunc* GetFn(std::string fn_name) const {
    std::string global_fn_name("device_api." + device_name_ + "." + fn_name);
    static const PackedFunc* f = Registry::Get(global_fn_name);
    CHECK(f != nullptr) << "Could not find function `" << global_fn_name << "`";
    return f;
  }
};

TVM_REGISTER_GLOBAL("device_api.create_ffi_api")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::string device_name = args[0];
    DeviceAPI* ptr = new FFIDeviceAPI(device_name);
    *rv = ptr;
  });

TVM_REGISTER_GLOBAL("device_api.destroy_ffi_api")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    void* ptr = args[0];
    delete reinterpret_cast<FFIDeviceAPI*>(ptr);
  });

}  // namespace runtime
}  // namespace tvm
