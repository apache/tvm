/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_module.cc
 */
#include <utility>

#include "./opengl_common.h"
#include "./opengl_module.h"

#if TVM_OPENGL_RUNTIME

#include <tvm/runtime/registry.h>
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "../meta_data.h"
#include "../file_util.h"

namespace tvm {
namespace runtime {

class OpenGLModuleNode final : public ModuleNode {
 public:
  explicit OpenGLModuleNode(std::string data,
                            std::string fmt,
                            std::unordered_map<std::string, FunctionInfo> fmap);

  // Disallow copy and assignment.
  OpenGLModuleNode(const OpenGLModuleNode& other) = delete;
  OpenGLModuleNode(OpenGLModuleNode&& other) = delete;
  OpenGLModuleNode& operator=(const OpenGLModuleNode& other) = delete;
  OpenGLModuleNode& operator=(OpenGLModuleNode&& other) = delete;

  ~OpenGLModuleNode() override = default;

  const char* type_key() const final { return "opengl"; }

  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  std::string GetSource(const std::string& format) final;

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    CHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    std::string meta_file = GetMetaFilePath(file_name);
    SaveMetaDataToFile(meta_file, fmap_);
    SaveBinaryToFile(file_name, data_);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

  const gl::Program &program() const { return program_; }

  std::shared_ptr<gl::OpenGLWorkspace> workspace_;

 private:
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
  gl::Program program_;
};

class OpenGLWrappedFunc {
 public:
  OpenGLWrappedFunc(OpenGLModuleNode *m,
                    std::shared_ptr<ModuleNode> sptr,
                    std::string func_name,
                    std::vector<size_t> arg_size,
                    const std::vector<std::string>& thread_axis_tags);

  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const;

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

OpenGLModuleNode::OpenGLModuleNode(
    std::string data, std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap)
    : workspace_(gl::OpenGLWorkspace::Global()), data_(std::move(data)),
      fmt_(std::move(fmt)), fmap_(std::move(fmap)),
      program_(workspace_->CreateProgram(data_.c_str())) {
  LOG(INFO) << "OpenGLModuleNode(" << data << ", " << fmt << ", "
                    << fmap.size() << ")";
  CHECK(fmt_ == "gl") << "Unknown OpenGL format " << fmt_;
}

PackedFunc OpenGLModuleNode::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  LOG(INFO) << "OpenGLModuleNode::GetFunction(" << name << ")";
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";

  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;

  std::vector<size_t> arg_size(info.arg_types.size());
  for (size_t i = 0; i < info.arg_types.size(); ++i) {
    TVMType t = info.arg_types[i];
    CHECK_EQ(t.lanes, 1U);
    uint32_t bits = t.bits;
    CHECK_EQ(bits % 8, 0U);
    arg_size[i] = bits / 8;
  }

  // initialize the wrapped func.
  OpenGLWrappedFunc f(this, sptr_to_self, name, arg_size,
                      info.thread_axis_tags);
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

OpenGLWrappedFunc::OpenGLWrappedFunc(
    OpenGLModuleNode* m,
    std::shared_ptr<ModuleNode> sptr,
    std::string func_name,
    std::vector<size_t> arg_size,
    const std::vector<std::string>& thread_axis_tags)
    : m_(m), sptr_(std::move(sptr)), func_name_(std::move(func_name)),
      arg_size_(std::move(arg_size)) {
  LOG(INFO) << "OpenGLWrappedFunc(" << func_name_ << ", "
            << "nargs = " << arg_size_.size() << ", "
            << "nthread_axis_tags = " << thread_axis_tags.size() << ")";
  for (auto& a : arg_size_) {
    LOG(INFO) << a;
  }
  for (auto& t : thread_axis_tags) {
    LOG(INFO) << t;
  }

  thread_axis_cfg_.Init(arg_size_.size(), thread_axis_tags);
}

void OpenGLWrappedFunc::operator()(TVMArgs args, TVMRetValue* rv,
                                   void** void_args) const {
  LOG(INFO) << "OpenGLWrappedFunc::operator()";

  gl::Texture* arg0 = *static_cast<gl::Texture**>(void_args[0]);
  std::vector<std::pair<std::string, gl::Texture*>> name_texture_pairs;
  for (size_t i = 1; i != arg_size_.size(); ++i) {
    name_texture_pairs.push_back(std::make_pair(
        "arg" + std::to_string(i),
        *static_cast<gl::Texture**>(void_args[i])
    ));
  }

  m_->workspace_->Render(
      m_->program(),
      name_texture_pairs,
      arg0
  );
}

Module OpenGLModuleCreate(std::string data,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap) {
  LOG(INFO) << "OpenGLModuleCreate()";
  auto n = std::make_shared<OpenGLModuleNode>(data, fmt, fmap);
  return Module(n);
}

Module OpenGLModuleLoadFile(const std::string& file_name,
                            const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return OpenGLModuleCreate(data, fmt, fmap);}

Module OpenGLModuleLoadBinary(void* strm) {
  auto stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return OpenGLModuleCreate(data, fmt, fmap);
}

TVM_REGISTER_GLOBAL("module.loadfile_gl")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = OpenGLModuleLoadFile(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("module.loadfile_glbin")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = OpenGLModuleLoadFile(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("module.loadbinary_opengl")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = OpenGLModuleLoadBinary(args[0]);
  });

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENGL_RUNTIME
