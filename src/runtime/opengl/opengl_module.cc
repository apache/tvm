/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_module.cc
 */
#include "./opengl_common.h"
#include "./opengl_module.h"

#if TVM_OPENGL_RUNTIME

namespace tvm {
namespace runtime {

class OpenGLModuleNode final : public ModuleNode {
 public:
  explicit OpenGLModuleNode(
      std::string data,
      std::string fmt,
      std::unordered_map<std::string, FunctionInfo> fmap) {
    // TODO(zhixunt): Implement this.
  }

  OpenGLModuleNode(const OpenGLModuleNode &other) = delete;
  OpenGLModuleNode(OpenGLModuleNode &&other) = delete;
  OpenGLModuleNode& operator=(const OpenGLModuleNode &other) = delete;
  OpenGLModuleNode& operator=(OpenGLModuleNode &&other) = delete;

  ~OpenGLModuleNode() override {
    // TODO(zhixunt): Implement this.
  }

  const char* type_key() const final {
    return "opengl";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  void Init() {
    // TODO(zhixunt): Implement this.
  }

};

PackedFunc OpenGLModuleNode::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // TODO(zhixunt): Implement this.
  throw "Not Implemented";
}

Module OpenGLModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap) {
  auto n = std::make_shared<OpenGLModuleNode>(data, fmt, fmap);
  n->Init();
  return Module(n);
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENGL_RUNTIME
