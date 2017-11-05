/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_device_api.cc
 */
#include "./opengl_common.h"

#if TVM_OPENGL_RUNTIME

namespace tvm {
namespace runtime {
namespace gl {

void OpenGLWorkspace::SetDevice(TVMContext ctx) {
  // TODO(zhixunt): Implement this.
}

void OpenGLWorkspace::GetAttr(
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue *rv) {
  // TODO(zhixunt): Implement this.
}

void* OpenGLWorkspace::AllocDataSpace(
    TVMContext ctx, size_t size, size_t alignment) {
  // TODO(zhixunt): Implement this.
  throw "Not Implemented";
}

void OpenGLWorkspace::FreeDataSpace(TVMContext ctx, void *ptr) {
  // TODO(zhixunt): Implement this.
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
}

void OpenGLWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  // TODO(zhixunt): Implement this.
}

void* OpenGLWorkspace::AllocWorkspace(TVMContext ctx, size_t size) {
  // TODO(zhixunt): Implement this.
  throw "Not Implemented";
}

void OpenGLWorkspace::FreeWorkspace(TVMContext ctx, void *data) {
  // TODO(zhixunt): Implement this.
}

}  // namespace gl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENGL_RUNTIME
