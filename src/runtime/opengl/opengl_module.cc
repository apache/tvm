/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_module.cc
 */
#include <tvm/runtime/registry.h>
#include <utility>
#include <unordered_map>
#include "opengl_common.h"
#include "opengl_module.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "../file_util.h"

namespace tvm {
namespace runtime {

class OpenGLModuleNode final : public ModuleNode {
 public:
  OpenGLModuleNode(std::unordered_map<std::string, OpenGLShader> shaders,
                   std::string fmt,
                   std::unordered_map<std::string, FunctionInfo> fmap);

  ~OpenGLModuleNode() override = default;

  const char* type_key() const final { return "opengl"; }

  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  std::string GetSource(const std::string& format) final;

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final;

  void SaveToBinary(dmlc::Stream* stream) final;

  const gl::Program& GetProgram(const std::string& func_name) const;

  const OpenGLShader& GetShader(const std::string& func_name) const;

  const FunctionInfo& GetFunctionInfo(const std::string& func_name) const;

  gl::OpenGLWorkspace& workspace() const { return *workspace_; }

 private:
  std::shared_ptr<gl::OpenGLWorkspace> workspace_;
  std::unordered_map<std::string, OpenGLShader> shaders_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
  std::unordered_map<std::string, gl::Program> programs_;

  DISALLOW_COPY_AND_ASSIGN(OpenGLModuleNode);
};

class OpenGLWrappedFunc {
 public:
  OpenGLWrappedFunc(OpenGLModuleNode* m,
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
    std::unordered_map<std::string, OpenGLShader> shaders,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap)
    : workspace_(gl::OpenGLWorkspace::Global()), shaders_(std::move(shaders)),
      fmt_(std::move(fmt)), fmap_(std::move(fmap)), programs_() {
  CHECK_EQ(fmt_, "gl") << "Unknown OpenGL format " << fmt_;
  for (auto &pair : shaders_) {
    auto &func_name = pair.first;
    auto &shader = pair.second;
    programs_.emplace(func_name,
                      workspace_->CreateProgram(shader.source.c_str()));
  }
}

PackedFunc OpenGLModuleNode::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";

  auto func_info_it = fmap_.find(name);
  if (func_info_it == fmap_.end()) { return PackedFunc(); }
  auto &func_info = func_info_it->second;

  std::vector<size_t> arg_size(func_info.arg_types.size());
  for (size_t i = 0; i < func_info.arg_types.size(); ++i) {
    TVMType t = func_info.arg_types[i];
    CHECK_EQ(t.lanes, 1U);
    uint32_t bits = t.bits;
    CHECK_EQ(bits % 8, 0U);
    arg_size[i] = bits / 8;
  }

  // Initialize the wrapped func.
  OpenGLWrappedFunc f(this, sptr_to_self, name, arg_size,
                      func_info.thread_axis_tags);
  return PackFuncVoidAddr(f, func_info.arg_types);
}

std::string OpenGLModuleNode::GetSource(const std::string& format) {
  if (format != fmt_ && fmt_ != "gl") { return ""; }

  std::ostringstream os;
  for (auto &pair : shaders_) {
    auto &name = pair.first;
    auto &shader = pair.second;
    os << "[" << name << "]" << "\n";
    os << shader.source <<"\n";
  }
  return os.str();
}

void OpenGLModuleNode::SaveToFile(const std::string& file_name,
                                  const std::string& format) {
  std::string fmt = GetFileFormat(file_name, format);
  CHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
  std::string meta_file = GetMetaFilePath(file_name);
  SaveMetaDataToFile(meta_file, fmap_);
  SaveBinaryToFile(file_name, ToJSON(shaders_));
}

void OpenGLModuleNode::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(fmt_);
  stream->Write(fmap_);
  stream->Write(ToJSON(shaders_));
}

const gl::Program& OpenGLModuleNode::GetProgram(
    const std::string& func_name) const {
  auto it = programs_.find(func_name);
  if (it == programs_.end()) {
    LOG(FATAL) << "Cannot find program";
  }
  return it->second;
}

const OpenGLShader& OpenGLModuleNode::GetShader(
    const std::string& func_name) const {
  auto it = shaders_.find(func_name);
  if (it == shaders_.end()) {
    LOG(FATAL) << "Cannot find shader";
  }
  return it->second;
}

const FunctionInfo& OpenGLModuleNode::GetFunctionInfo(
    const std::string& func_name) const {
  auto it = fmap_.find(func_name);
  if (it == fmap_.end()) {
    LOG(FATAL) << "Cannot find shader";
  }
  return it->second;
}

OpenGLWrappedFunc::OpenGLWrappedFunc(
    OpenGLModuleNode* m,
    std::shared_ptr<ModuleNode> sptr,
    std::string func_name,
    std::vector<size_t> arg_size,
    const std::vector<std::string>& thread_axis_tags)
    : m_(m), sptr_(std::move(sptr)), func_name_(std::move(func_name)),
      arg_size_(std::move(arg_size)) {
  thread_axis_cfg_.Init(arg_size_.size(), thread_axis_tags);
}

void OpenGLWrappedFunc::operator()(TVMArgs args, TVMRetValue* rv,
                                   void** void_args) const {
  auto &shader = m_->GetShader(func_name_);
  auto &program = m_->GetProgram(func_name_);
  auto &func_info = m_->GetFunctionInfo(func_name_);
  size_t nargs = shader.arg_kinds.size();

  // Must call this function before setting uniforms & input textures.
  m_->workspace().SetCurrentProgram(program);

  // Set all arguments.
  GLuint texture_unit = 0;
  gl::Texture* output = nullptr;
  for (size_t i = 0; i != nargs; ++i) {
    auto &name = shader.arg_names.at(i);
    auto kind = shader.arg_kinds.at(i);
    auto type = func_info.arg_types.at(i);
    switch (kind) {
      case OpenGLArgKind::kUniform: {
        m_->workspace().SetUniform(program, name, type, void_args[i]);
        break;
      }
      case OpenGLArgKind::kInputTexture: {
        CHECK_EQ(type.code, kHandle) << "Type is not handle?";
        auto texture = *static_cast<gl::Texture**>(void_args[i]);
        m_->workspace().SetInputTexture(program, name, texture_unit, texture);
        ++texture_unit;
        break;
      }
      case OpenGLArgKind::kOutputTexture: {
        CHECK_EQ(type.code, kHandle) << "Type is not handle?";
        CHECK(output == nullptr) << "Can only have one output texture.";
        output = *static_cast<gl::Texture**>(void_args[i]);
        break;
      }
    }
  }

  // Set "thread_extent" uniform.
  ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
  std::unique_ptr<GLint> thread_extent(new GLint(wl.block_dim(0)));
  m_->workspace().SetUniform(program, shader.thread_extent_var,
                             TVMType{kDLInt, 32, 1},
                             static_cast<void*>(thread_extent.get()));

  m_->workspace().Render(output);
}

Module OpenGLModuleCreate(std::unordered_map<std::string, OpenGLShader> shaders,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap) {
  auto n = std::make_shared<OpenGLModuleNode>(std::move(shaders),
                                              std::move(fmt),
                                              std::move(fmap));
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
  return OpenGLModuleCreate(FromJSON(data), fmt, fmap);
}

Module OpenGLModuleLoadBinary(void* strm) {
  auto stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return OpenGLModuleCreate(FromJSON(data), fmt, fmap);
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
