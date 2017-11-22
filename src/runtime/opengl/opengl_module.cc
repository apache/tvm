/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_module.cc
 */
#include <utility>

#include "./opengl_common.h"
#include "./opengl_module.h"

#if TVM_OPENGL_RUNTIME

#include "../pack_args.h"

namespace tvm {
namespace runtime {

class OpenGLModuleNode final : public ModuleNode {
 public:
  explicit OpenGLModuleNode(std::string data,
                            std::string fmt,
                            std::unordered_map<std::string, FunctionInfo> fmap)
      : data_(std::move(data)), fmt_(std::move(fmt)), fmap_(std::move(fmap)) {
    // TODO(zhixunt): Implement this.
  }

  OpenGLModuleNode(const OpenGLModuleNode &other) = delete;
  OpenGLModuleNode(OpenGLModuleNode &&other) = delete;
  OpenGLModuleNode& operator=(const OpenGLModuleNode &other) = delete;
  OpenGLModuleNode& operator=(OpenGLModuleNode &&other) = delete;

  ~OpenGLModuleNode() override = default;

  const char* type_key() const final { return "opengl"; }

  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  std::string GetSource(const std::string& format) final;

 private:
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
};

class OpenGLWrappedFunc {
 public:
  void operator()(TVMArgs args, TVMRetValue *rv, void **void_args) const {
    // TODO(zhixunt): Implement this.
    LOG_INFO.stream() << "OpenGLWrappedFunc::operator()";
  }
};

PackedFunc OpenGLModuleNode::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "OpenGLModuleNode::GetFunction";
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main)
    << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo &info = it->second;
  OpenGLWrappedFunc f;
  return PackFuncVoidAddr(f, info.arg_types);
}

std::string OpenGLModuleNode::GetSource(const std::string& format) {
  if (format == fmt_) return data_;
  if (fmt_ == "gl") {
    return data_;
  } else {
    return "";
  }
}

Module OpenGLModuleCreate(std::string data,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap) {
  auto n = std::make_shared<OpenGLModuleNode>(data, fmt, fmap);
  return Module(n);
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENGL_RUNTIME
