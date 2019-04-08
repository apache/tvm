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
 * \file codegen_opengl.h
 * \brief Generate OpenGL device code.
 */
#ifndef TVM_CODEGEN_CODEGEN_OPENGL_H_
#define TVM_CODEGEN_CODEGEN_OPENGL_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include "codegen_c.h"
#include "../runtime/opengl/opengl_module.h"

namespace tvm {
namespace codegen {

class CodeGenOpenGL final : public CodeGenC {
 public:
  CodeGenOpenGL();
  void AddFunction(LoweredFunc f);
  std::unordered_map<std::string, runtime::OpenGLShader> Finish();

  void InitFuncState(LoweredFunc f) final;
  void BindThreadIndex(const IterVar& iv) final;
  void VisitStmt_(const Store* op) final;
  std::string TexelFetch(const Variable* buffer, Expr index);
  std::string GetBufferRef(Type t, const Variable* buffer, Expr index) final;
  void PrintType(Type t, std::ostream& os) final; // NOLINT(*)

  // Codegen for immediate values
  void VisitExpr_(const IntImm* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const UIntImm* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const FloatImm* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const StringImm* op, std::ostream& os) final;  // NOLINT(*)

  // Match glsl_texture_store Call.
  void VisitStmt_(const Evaluate* op) final;  // NOLINT(*)

 private:
  const Variable* output_{nullptr};
  std::unordered_set<const Variable*> inputs_;
  const Variable* output_iter_var_{nullptr};
  std::unordered_map<std::string, runtime::OpenGLShader> shaders_;
  std::string thread_extent_var_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_OPENGL_H_
