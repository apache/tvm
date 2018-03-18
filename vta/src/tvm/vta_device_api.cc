// simply include the driver for now.
#include <tvm/runtime/registry.h>
#include <dmlc/thread_local.h>
#include <vta/runtime.h>
#include "../../tvm/src/runtime/workspace_pool.h"

namespace tvm {
namespace runtime {

std::string VTARPCGetPath(const std::string& name) {
  static const PackedFunc* f =
      runtime::Registry::Get("tvm.contrib.rpc.server.workpath");
  CHECK(f != nullptr) << "require tvm.contrib.rpc.server.workpath";
  return (*f)(name);
}

// Global functions that can be called
TVM_REGISTER_GLOBAL("tvm.contrib.vta.init")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::string path = VTARPCGetPath(args[0]);
    VTAProgram(path.c_str());
    LOG(INFO) << "VTA initialization end with bistream " << path;
  });

TVM_REGISTER_GLOBAL("tvm.contrib.rpc.server.shutdown")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    VTARuntimeShutdown();
  });

class VTADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {}

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void* AllocDataSpace(TVMContext ctx,
                       size_t size, size_t alignment,
                       TVMType type_hint) final {
    return VTABufferAlloc(VTATLSCommandHandle(), size);
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    VTABufferFree(VTATLSCommandHandle(), ptr);
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMStreamHandle stream) final {
    int kind_mask = 0;
    if (ctx_from.device_type != kDLCPU) {
      kind_mask |= 2;
    }
    if (ctx_to.device_type != kDLCPU) {
      kind_mask |= 1;
    }
    VTABufferCopy(VTATLSCommandHandle(),
                  from, from_offset,
                  to, to_offset,
                  size, kind_mask);
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final;

  void FreeWorkspace(TVMContext ctx, void* data) final;

  static const std::shared_ptr<VTADeviceAPI>& Global() {
    static std::shared_ptr<VTADeviceAPI> inst =
        std::make_shared<VTADeviceAPI>();
    return inst;
  }
};

struct VTAWorkspacePool : public WorkspacePool {
  VTAWorkspacePool() :
      WorkspacePool(static_cast<DLDeviceType>(kExtDev),
                    VTADeviceAPI::Global()) {}
};

void* VTADeviceAPI::AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) {
  return dmlc::ThreadLocalStore<VTAWorkspacePool>::Get()
      ->AllocWorkspace(ctx, size);
}

void VTADeviceAPI::FreeWorkspace(TVMContext ctx, void* data) {
  dmlc::ThreadLocalStore<VTAWorkspacePool>::Get()->FreeWorkspace(ctx, data);
}

TVM_REGISTER_GLOBAL("device_api.ext_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = VTADeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });
}  // namespace runtime
}  // namespace tvm
