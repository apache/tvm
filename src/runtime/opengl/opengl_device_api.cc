/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_device_api.cc
 */
#include "./opengl_common.h"

#if TVM_OPENGL_RUNTIME

#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
namespace gl {

const std::shared_ptr<OpenGLWorkspace>& OpenGLWorkspace::Global() {
  static std::shared_ptr<OpenGLWorkspace> inst = std::make_shared<OpenGLWorkspace>();
  return inst;
}

void OpenGLWorkspace::SetDevice(TVMContext ctx) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::SetDevice";
}

void OpenGLWorkspace::GetAttr(
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue *rv) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::GetAttr";
}

void* OpenGLWorkspace::AllocDataSpace(
    TVMContext ctx, size_t size, size_t alignment) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream()
      << "OpenGLWorkspace::AllocDataSpace(ctx, size = "
      << size << ", alignment = " << alignment << ")";
  return nullptr;
}

void OpenGLWorkspace::FreeDataSpace(TVMContext ctx, void *ptr) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream()
      << "OpenGLWorkspace::FreeDataSpace(ctx, ptr = "
      << ptr << ")";
}

void OpenGLWorkspace::CopyDataFromTo(const void *from,
                                     size_t from_offset,
                                     void *to,
                                     size_t to_offset,
                                     size_t size,
                                     TVMContext ctx_from,
                                     TVMContext ctx_to,
                                     TVMStreamHandle stream) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream()
      << "OpenGLWorkspace::CopyDataFromTo("
      << "from = " << from << ", "
      << "from_offset = " << from_offset << ", "
      << "to = " << to << ", "
      << "to_offset = " << to_offset << ", "
      << "size = " << size << ", "
      << "ctx_from, ctx_to, stream)";
}

void OpenGLWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::StreamSync";
}

void* OpenGLWorkspace::AllocWorkspace(TVMContext ctx, size_t size) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::AllocWorkspace";
  return nullptr;
}

void OpenGLWorkspace::FreeWorkspace(TVMContext ctx, void *data) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLWorkspace::FreeWorkspace";
}

TVM_REGISTER_GLOBAL("device_api.opengl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = OpenGLWorkspace::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace gl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENGL_RUNTIME
