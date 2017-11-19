/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_common.h
 * \brief OpenGL common header
 */
#ifndef TVM_RUNTIME_OPENGL_OPENGL_COMMON_H_
#define TVM_RUNTIME_OPENGL_OPENGL_COMMON_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <dmlc/logging.h>

namespace tvm {
namespace runtime {
namespace gl {

/*!
 * \brief Process global OpenGL workspace.
 */
class OpenGLWorkspace final : public DeviceAPI {
 public:
  void Init();

  // override device API
  void SetDevice(TVMContext ctx) final;
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment) final;
  void FreeDataSpace(TVMContext ctx, void* ptr) final;
  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMStreamHandle stream) final;
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final;
  void* AllocWorkspace(TVMContext ctx, size_t size) final;
  void FreeWorkspace(TVMContext ctx, void* data) final;
  // get the global workspace
  static const std::shared_ptr<OpenGLWorkspace>& Global();
};

}  // namespace gl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_OPENGL_OPENGL_COMMON_H_
