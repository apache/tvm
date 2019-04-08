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
 * \file opengl_module.h
 * \brief Execution handling of OpenGL kernels
 */
#ifndef TVM_RUNTIME_OPENGL_OPENGL_MODULE_H_
#define TVM_RUNTIME_OPENGL_OPENGL_MODULE_H_

#include <tvm/runtime/packed_func.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include "../meta_data.h"

namespace tvm {
namespace runtime {

/*!
 * \brief The fixed row size of all OpenGL textures in TVM.
 *
 * OpenGL has texture size limit on each dimension. Suppose we have a limit of
 * 1024, then we can have a 2D texture of size (2^10 x 2^10) but not (2^20 x 1).
 * This means we don't want to just use (n x 1) 2D textures for all arrays,
 * because that would limit our array size to be 1024. Here we use (1024 x m)
 * 2D textures. Then we can have arrays of size up to 2^20.
 */
static constexpr int kTextureRowBits = 10;
static constexpr int kTextureRowSize = 1 << kTextureRowBits;
static constexpr int kTextureRowMask = kTextureRowSize - 1;

/*!
 * \brief Determines how we supply arguments.
 */
enum class OpenGLArgKind {
  kInputTexture = 0,   // Bind to "gsampler2D" in GLSL.
  kOutputTexture = 1,  // Bind to "out" in GLSL.
  kUniform = 2,        // Bind to "uniform" in GLSL.
};

std::string OpenGLArgKind2String(OpenGLArgKind kind);
OpenGLArgKind String2OpenGLArgKind(const std::string& str);

/*!
 * \brief The output of OpenGL codegen.
 * Contains necessary information to build a fragment shader and bind arguments.
 */
struct OpenGLShader {
  OpenGLShader() = default;
  OpenGLShader(std::string source,
               std::vector<std::string> arg_names,
               std::vector<OpenGLArgKind> arg_kinds,
               std::string thread_extent_var)
      : source(std::move(source)), arg_names(std::move(arg_names)),
        arg_kinds(std::move(arg_kinds)),
        thread_extent_var(std::move(thread_extent_var)) {
    CHECK_EQ(this->arg_names.size(), this->arg_kinds.size()) << "Invalid input";
  }

  std::string source;
  std::vector<std::string> arg_names;    // Matches FunctionInfo.
  std::vector<OpenGLArgKind> arg_kinds;  // Matches FunctionInfo.
  std::string thread_extent_var;         // Stores the output length.

  void Save(dmlc::JSONWriter* writer) const;
  void Load(dmlc::JSONReader* reader);
};

std::string ToJSON(const std::unordered_map<std::string, OpenGLShader>& shaders);
std::unordered_map<std::string, OpenGLShader> FromJSON(const std::string& str);

/*!
 * \brief Create an OpenGL module from data.
 *
 * \param data The module data.
 * \param fmt The format of the data,
 * \param fmap The map function information map of each function.
 */
Module OpenGLModuleCreate(std::unordered_map<std::string, OpenGLShader> shaders,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap);

inline std::string OpenGLArgKind2String(OpenGLArgKind kind) {
  switch (kind) {
    case OpenGLArgKind::kOutputTexture:
      return "output_texture";
    case OpenGLArgKind::kInputTexture:
      return "input_texture";
    case OpenGLArgKind::kUniform:
      return "uniform";
    default:
      LOG(FATAL) << "invalid arg kind";
      return "";
  }
}

inline OpenGLArgKind String2OpenGLArgKind(const std::string& str) {
  if (str == "output_texture") {
    return OpenGLArgKind::kOutputTexture;
  } else if (str == "input_texture") {
    return OpenGLArgKind::kInputTexture;
  } else if (str == "uniform") {
    return OpenGLArgKind::kUniform;
  } else {
    LOG(FATAL) << "Invalid OpenGL arg kind.";
    return OpenGLArgKind::kUniform;
  }
}

inline void OpenGLShader::Save(dmlc::JSONWriter* writer) const {
  std::vector<std::string> arg_kind_strs;
  for (auto kind : arg_kinds) {
    arg_kind_strs.push_back(OpenGLArgKind2String(kind));
  }

  writer->BeginObject();
  writer->WriteObjectKeyValue("arg_names", arg_names);
  writer->WriteObjectKeyValue("arg_kinds", arg_kind_strs);
  writer->WriteObjectKeyValue("source", source);
  writer->WriteObjectKeyValue("thread_extent_var", thread_extent_var);
  writer->EndObject();
}

inline void OpenGLShader::Load(dmlc::JSONReader* reader) {
  std::vector<std::string> arg_kind_strs;
  dmlc::JSONObjectReadHelper helper;
  helper.DeclareField("arg_names", &arg_names);
  helper.DeclareField("arg_kinds", &arg_kind_strs);
  helper.DeclareField("source", &source);
  helper.DeclareField("thread_extent_var", &thread_extent_var);
  helper.ReadAllFields(reader);

  arg_kinds.clear();
  for (auto& str : arg_kind_strs) {
    arg_kinds.push_back(String2OpenGLArgKind(str));
  }
}

inline std::string ToJSON(
    const std::unordered_map<std::string, OpenGLShader>& shaders) {
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  writer.BeginObject();
  writer.WriteObjectKeyValue("shaders", shaders);
  writer.EndObject();
  return os.str();
}

inline std::unordered_map<std::string, OpenGLShader> FromJSON(
    const std::string& str) {
  std::unordered_map<std::string, OpenGLShader> shaders;
  std::istringstream is(str);
  dmlc::JSONReader reader(&is);
  dmlc::JSONObjectReadHelper helper;
  helper.DeclareField("shaders", &shaders);
  helper.ReadAllFields(&reader);
  return shaders;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENGL_OPENGL_MODULE_H_
