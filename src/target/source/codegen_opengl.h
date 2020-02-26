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
 * \file codegen_opengl.h
 * \brief Generate OpenGL device code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_OPENGL_H_
#define TVM_TARGET_SOURCE_CODEGEN_OPENGL_H_

#include <tvm/target/codegen.h>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include "codegen_c.h"
#include "../../runtime/opengl/opengl_module.h"

namespace tvm {
namespace codegen {

class CodeGenOpenGL final : public CodeGenC {
 public:
  CodeGenOpenGL();
  void AddFunction(LoweredFunc f);
  std::unordered_map<std::string, runtime::OpenGLShader> Finish();

  void InitFuncState(LoweredFunc f) final;
  void BindThreadIndex(const IterVar& iv) final;
  void VisitStmt_(const StoreNode* op) final;
  std::string TexelFetch(const VarNode* buffer, PrimExpr index);
  std::string GetBufferRef(DataType t, const VarNode* buffer, PrimExpr index) final;
  void PrintType(DataType t, std::ostream& os) final; // NOLINT(*)

  // Codegen for immediate values
  void VisitExpr_(const IntImmNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const StringImmNode* op, std::ostream& os) final;  // NOLINT(*)

  // Match glsl_texture_store Call.
  void VisitStmt_(const EvaluateNode* op) final;  // NOLINT(*)

 private:
  const VarNode* output_{nullptr};
  std::unordered_set<const VarNode*> inputs_;
  const VarNode* output_iter_var_{nullptr};
  std::unordered_map<std::string, runtime::OpenGLShader> shaders_;
  std::string thread_extent_var_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_OPENGL_H_
