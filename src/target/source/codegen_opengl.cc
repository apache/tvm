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
 * \file codegen_opengl.cc
 *
 * We are targeting OpenGL 3.3. The reason of not targeting a recent version
 * of OpenGL is to have better compatibility of WebGL 2.
 */
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include "codegen_opengl.h"
#include "../build_common.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace codegen {

CodeGenOpenGL::CodeGenOpenGL()
    : output_(nullptr), output_iter_var_(nullptr) {}

void CodeGenOpenGL::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  output_ = nullptr;
  inputs_.clear();
  output_iter_var_ = nullptr;
  thread_extent_var_ = "";
  this->decl_stream.str("");
  this->stream.str("");
}

void CodeGenOpenGL::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);

  this->decl_stream << "#version 300 es\n";
  this->decl_stream << "precision highp float;\n";

  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");

  // Allocate argument names. Store in `var_idmap_`.
  for (auto arg : f->params) {
    auto arg_name = GetUniqueName(arg.get()->name_hint);
    var_idmap_[arg.get()] = arg_name;

    if (auto* ptr = arg->type_annotation.as<PointerTypeNode>()) {
      if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(arg.get(), prim->dtype);
      }
    }
  }

  thread_extent_var_ = GetUniqueName("thread_extent");
  this->decl_stream << "uniform int " << thread_extent_var_ << ";\n";

  this->stream << "void main() {\n";

  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);

  this->PrintIndent();
  this->stream << "}\n\n";

  // Declare arguments.
  for (auto arg : f->params) {
    if (this->inputs_.find(arg.get()) != this->inputs_.cend()) {
      // Declare input texture.
      // Format:
      // - Float: "uniform sampler2D {name};"
      // - Int: "uniform isampler2D {name};"
      // - UInt: "uniform usampler2D {name};"

      auto arg_name = GetVarID(arg.get());

      auto type_it = this->handle_data_type_.find(arg.get());
      CHECK(type_it != this->handle_data_type_.cend()) << "Cannot find type.";
      DLDataType type = type_it->second;
      CHECK_EQ(type.lanes, 1) << "Vector type not supported.";

      switch (type.code) {
        case kDLInt:
          this->decl_stream << "uniform isampler2D " << arg_name << ";\n";
          break;
        case kDLUInt:
          this->decl_stream << "uniform usampler2D " << arg_name << ";\n";
          break;
        case kDLFloat:
          this->decl_stream << "uniform sampler2D " << arg_name << ";\n";
          break;
        default:
          LOG(FATAL) << "Unsupported type code.";
      }

    } else if (this->output_ == arg.get()) {
      // Declare output texture.
      // Format: "out {type} {name};"

      auto arg_name = GetVarID(arg.get());

      auto type_it = this->handle_data_type_.find(arg.get());
      CHECK(type_it != this->handle_data_type_.cend()) << "Cannot find type.";
      auto type = type_it->second;

      this->decl_stream << "out ";
      PrintType(type, this->decl_stream);
      this->decl_stream << " " << arg_name << ";\n";

    } else {
      // Declare uniform value.
      // Format: "uniform {type} {name};"

      auto arg_name = GetVarID(arg.get());
      auto type = arg.get()->dtype;

      this->decl_stream << "uniform ";
      PrintType(type, this->decl_stream);
      this->decl_stream << " " << arg_name << ";\n";
    }
  }

  std::vector<std::string> arg_names;
  std::vector<runtime::OpenGLArgKind> arg_kinds;
  for (auto arg : f->params) {
    std::string name = GetVarID(arg.get());

    runtime::OpenGLArgKind kind;
    if (inputs_.find(arg.get()) != inputs_.cend()) {
      kind = runtime::OpenGLArgKind::kInputTexture;
    } else if (output_ == arg.get()) {
      kind = runtime::OpenGLArgKind::kOutputTexture;
    } else {
      kind = runtime::OpenGLArgKind::kUniform;
    }

    arg_names.push_back(name);
    arg_kinds.push_back(kind);
  }

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  CHECK(global_symbol.defined())
      << "CodeGenOpenGL: Expect PrimFunc to have the global_symbol attribute";

  shaders_[static_cast<std::string>(global_symbol.value())] = runtime::OpenGLShader(
      this->decl_stream.str() + this->stream.str(),
      std::move(arg_names), std::move(arg_kinds),
      this->thread_extent_var_);
}

std::unordered_map<std::string, runtime::OpenGLShader> CodeGenOpenGL::Finish() {
  return shaders_;
}

void CodeGenOpenGL::BindThreadIndex(const IterVar& iv) {
  CHECK_EQ(iv->thread_tag, "threadIdx.x") << "Must be threadIdx.x";
  CHECK(var_idmap_.find(iv->var.get()) == var_idmap_.end())
    << "Only support one thread iter var";
  CHECK(output_iter_var_ == nullptr) << "Only support one thread iter var";

  var_idmap_[iv->var.get()] = iv->thread_tag;
  output_iter_var_ = iv->var.get();

  // Declare threadIdx local variable.
  this->PrintIndent();
  this->stream << "ivec2 threadIdx = ivec2(" << runtime::kTextureRowSize
               << " * int(gl_FragCoord.y) + int(gl_FragCoord.x), 0);\n";

  // Return directly if threadIdx.x >= thread_extent.
  this->PrintIndent();
  this->stream << "if (threadIdx.x >= " << thread_extent_var_ << ") {\n";
  this->PrintIndent();
  this->stream << "  return;\n";
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenOpenGL::VisitStmt_(const StoreNode* op) {
  LOG(FATAL) << "Store statement not supported in OpenGL."
             << " Texture store should be a Call statement.";
}

// texelFetch(tex, ivec2(idx & kTextureRowMask, idx >> kTextureRowBits), 0).r
std::string CodeGenOpenGL::TexelFetch(const VarNode* buffer, PrimExpr index) {
  std::ostringstream os;
  os << "texelFetch(" << GetVarID(buffer) << ", ivec2(int(";
  PrintExpr(index, os);
  os << ") & " << runtime::kTextureRowMask << ", int(";
  PrintExpr(index, os);
  os << ") >> " << runtime::kTextureRowBits << "), 0).r";
  return os.str();
}

// Print a reference expression to a buffer.
// Format: texelFetch(buffer, index, 0).r
std::string CodeGenOpenGL::GetBufferRef(
    DataType t, const VarNode* buffer, PrimExpr index) {
  CHECK_EQ(t.lanes(), 1) << "Vector type not supported.";
  CHECK(HandleTypeMatch(buffer, t)) << "Type mismatch not supported.";

  if (buffer == this->output_) {
    // This is the output texture.
    return GetVarID(buffer);
  } else {
    // This is an input texture.
    this->inputs_.insert(buffer);
    return TexelFetch(buffer, index);
  }
}

void CodeGenOpenGL::PrintType(DataType t, std::ostream& os) {
  switch (t.code()) {
    case kDLInt:
      CHECK_EQ(t.bits(), 32) << "Only support 32-bit int.";
      os << "int";
      break;
    case kDLUInt:
      CHECK_EQ(t.bits(), 32) << "Only support 32-bit uint.";
      os << "uint";
      break;
    case kDLFloat:
      CHECK_EQ(t.bits(), 32) << "Only support 32-bit float.";
      os << "float";
      break;
    default:
      LOG(FATAL) << "Unsupported type code.";
  }
}

// Codegen for immediate values

void CodeGenOpenGL::VisitExpr_(const IntImmNode* op, std::ostream& os) {
  CHECK_EQ(op->dtype, DataType::Int(32)) << "GLSL 3.0 only supports 32-bit ints.";
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenOpenGL::VisitExpr_(const FloatImmNode* op, std::ostream& os) {
  CHECK_EQ(op->dtype, DataType::Float(32)) << "GLSL 3.0 only supports 32-bit floats.";
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenOpenGL::VisitExpr_(const StringImmNode*, std::ostream& os) {
  LOG(FATAL) << "GLSL 3.0 doesn't support strings.";
}

void CodeGenOpenGL::VisitStmt_(const EvaluateNode* op) {
  auto call = op->value.as<CallNode>();
  if (call == nullptr || call->name != CallNode::glsl_texture_store) {
    // Fallback to normal logic.
    CodeGenC::VisitStmt_(op);
  }

  CHECK_EQ(call->args.size(), 2);
  auto buffer = call->args[0].as<VarNode>();
  auto value = call->args[1];

  // Doesn't support store to vector.
  auto type = value.dtype();
  CHECK_EQ(type.lanes(), 1)
    << "Vectorized store not implemented, type = " << type;

  CHECK(inputs_.find(buffer) == inputs_.cend())
    << "Texture has been read from before. Must not store to it.";
  if (output_ == nullptr) {
    output_ = buffer;  // Record that this texture is the output.
  } else {
    CHECK(output_ == buffer) << "GLSL can only write to 1 texture.";
  }

  this->PrintIndent();
  this->stream << GetVarID(buffer) << " = " << PrintExpr(value) << ";\n";
}

runtime::Module BuildOpenGL(IRModule mod) {
  bool output_ssa = false;
  CodeGenOpenGL cg;
  cg.Init(output_ssa);

  for (auto kv :  mod->functions) {
    CHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenOpenGL: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    CHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenOpenGL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
  }

  auto shaders = cg.Finish();
  return OpenGLModuleCreate(shaders, "gl", ExtractFuncInfo(mod));
}

TVM_REGISTER_GLOBAL("target.build.opengl")
.set_body_typed(BuildOpenGL);

}  // namespace codegen
}  // namespace tvm
