/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_module.h
 * \brief Execution handling of OpenGL kernels
 */
#ifndef TVM_RUNTIME_OPENGL_OPENGL_MODULE_H_
#define TVM_RUNTIME_OPENGL_OPENGL_MODULE_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/packed_func.h>
#include <memory>
#include <vector>
#include <string>
#include "../meta_data.h"

namespace tvm {
namespace runtime {

enum OpenGLArgKind : int {
  kInputTexture = 0,
  kOutputTexture = 1,
  kUniform = 2,
};

struct OpenGLShader {
  std::string source;
  std::vector<std::string> arg_names;
  std::vector<int> arg_kinds;

  void Save(dmlc::JSONWriter *writer) const;
  void Load(dmlc::JSONReader *reader);
};

std::string ToJson(
    const std::unordered_map<std::string, OpenGLShader>& shaders);

std::unordered_map<std::string, OpenGLShader> FromJson(const std::string& str);

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

inline void OpenGLShader::Save(dmlc::JSONWriter* writer) const {
  writer->BeginObject();
  writer->WriteObjectKeyValue("arg_names", arg_names);
  writer->WriteObjectKeyValue("arg_kinds", arg_kinds);
  writer->WriteObjectKeyValue("source", source);
  writer->EndObject();
}

inline void OpenGLShader::Load(dmlc::JSONReader* reader) {
  dmlc::JSONObjectReadHelper helper;
  helper.DeclareField("arg_names", &arg_names);
  helper.DeclareField("arg_kinds", &arg_kinds);
  helper.DeclareField("source", &source);
  helper.ReadAllFields(reader);
}

inline std::string ToJson(
    const std::unordered_map<std::string, OpenGLShader>& shaders) {
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  writer.BeginObject();
  writer.WriteObjectKeyValue("shaders", shaders);
  writer.EndObject();
  return os.str();
}

inline std::unordered_map<std::string, OpenGLShader> FromJson(
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
