/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_module.cc
 */
#include <utility>

#include "./opengl_common.h"
#include "./opengl_module.h"

#if TVM_OPENGL_RUNTIME

#include "../pack_args.h"
#include "../thread_storage_scope.h"

namespace tvm {
namespace runtime {

class OpenGLModuleNode final : public ModuleNode {
 public:
  explicit OpenGLModuleNode(std::string data,
                            std::string fmt,
                            std::unordered_map<std::string, FunctionInfo> fmap)
      : data_(std::move(data)), fmt_(std::move(fmt)), fmap_(std::move(fmap)) {
    LOG_INFO.stream() << data << " " << fmt << " " << fmap.size();
    for (auto &f: fmap) LOG_INFO.stream() << f.first;
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
  void Init() {
      LOG_INFO.stream() << fmt_ << " " << data_;
      workspace_ = gl::OpenGLWorkspace::Global();
      workspace_->Init();
      if (fmt_ == "gl") {
          const char* s = data_.c_str();
          program_ = workspace_->CreateProgram(s);
      } else {
          LOG(FATAL) << "Unknown OpenGL format " << fmt_;
      }
  }
    std::shared_ptr<gl::Program> program_;
    std::shared_ptr<gl::OpenGLWorkspace> workspace_;
 private:
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
};

class OpenGLWrappedFunc {
 public:
    void Init(OpenGLModuleNode* m,
              std::shared_ptr<ModuleNode> sptr,
              const std::string &func_name,
              std::vector<size_t> arg_size,
              const std::vector<std::string>& thread_axis_tags)  {
      LOG_INFO.stream() << func_name << " " << arg_size.size() << " " << thread_axis_tags.size();
      for (auto &a: arg_size) LOG_INFO.stream() << a;
      for (auto &t: thread_axis_tags) LOG_INFO.stream() << t;
      m_ = m;
      sptr_ = std::move(sptr);
      func_name_ = func_name;
      arg_size_ = arg_size;
      thread_axis_cfg_.Init(arg_size.size(), thread_axis_tags);
    }

  void operator()(TVMArgs args, TVMRetValue *rv, void **void_args) const {
    LOG_INFO.stream() << "OpenGLWrappedFunc::operator()";
      // TODO(pengw): How to get variable names?
      m_->workspace_->Render(*m_->program_, {
                                     {"A", *static_cast<gl::Texture**>(void_args[1])},
                                     {"B", *static_cast<gl::Texture**>(void_args[2])}
                             },
                             *static_cast<gl::Texture**>(void_args[0]));
  }

private:
    // The module
    OpenGLModuleNode* m_;
    // resource handle
    std::shared_ptr<ModuleNode> sptr_;
    // The name of the function.
    std::string func_name_;
    // convert code for void argument
    std::vector<size_t> arg_size_;
    // thread axis config
    ThreadAxisConfig thread_axis_cfg_;
};

PackedFunc OpenGLModuleNode::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  LOG_INFO.stream() << "OpenGLModuleNode::GetFunction" << " " << name;
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main)
    << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo &info = it->second;
  OpenGLWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());
  for (size_t i = 0; i < info.arg_types.size(); ++i) {
    TVMType t = info.arg_types[i];
    CHECK_EQ(t.lanes, 1U);
    uint32_t bits = t.bits;
    CHECK_EQ(bits % 8, 0U);
    arg_size[i] = bits / 8;
  }
  // initialize the wrapped func.
  f.Init(this, sptr_to_self,
         name, arg_size, info.thread_axis_tags);
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
    LOG_INFO.stream() << data << " " << fmt << " " << fmap.size();
    for (auto &f: fmap) LOG_INFO.stream() << f.first;
    n->Init();
  return Module(n);
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENGL_RUNTIME
