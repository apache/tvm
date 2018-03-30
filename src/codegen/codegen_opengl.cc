/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_opengl.cc
 *
 * We are targeting OpenGL 3.3. The reason of not targeting a recent version
 * of OpenGL is to have better compatibility of WebGL 2.
 */
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "./codegen_opengl.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace codegen {

CodeGenOpenGL::CodeGenOpenGL()
    : output_(nullptr), output_iter_var_(nullptr) {}

void CodeGenOpenGL::InitFuncState(LoweredFunc f) {
  CodeGenC::InitFuncState(f);
  output_ = nullptr;
  inputs_.clear();
  output_iter_var_ = nullptr;
  thread_extent_var_ = "";
  this->decl_stream.str("");
  this->stream.str("");
}

void CodeGenOpenGL::AddFunction(LoweredFunc f) {
  // clear previous generated state.
  this->InitFuncState(f);

  this->decl_stream << "#version 300 es\n";
  this->decl_stream << "precision highp float;\n";

  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");
  // add to alloc buffer type.
  for (const auto& kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  // Allocate argument names. Store in `var_idmap_`.
  for (auto arg : f->args) {
    auto arg_name = GetUniqueName(arg.get()->name_hint);
    var_idmap_[arg.get()] = arg_name;
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
  for (auto arg : f->args) {
    if (this->inputs_.find(arg.get()) != this->inputs_.cend()) {
      // Declare input texture.
      // Format:
      // - Float: "uniform sampler2D {name};"
      // - Int: "uniform isampler2D {name};"
      // - UInt: "uniform usampler2D {name};"

      auto arg_name = GetVarID(arg.get());

      auto type_it = this->handle_data_type_.find(arg.get());
      CHECK(type_it != this->handle_data_type_.cend()) << "Cannot find type.";
      auto type = Type2TVMType(type_it->second);
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
      auto type = arg.get()->type;

      this->decl_stream << "uniform ";
      PrintType(type, this->decl_stream);
      this->decl_stream << " " << arg_name << ";\n";
    }
  }

  std::vector<std::string> arg_names;
  std::vector<runtime::OpenGLArgKind> arg_kinds;
  for (auto arg : f->args) {
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

  shaders_[f->name] = runtime::OpenGLShader(
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

void CodeGenOpenGL::VisitStmt_(const Store* op) {
  LOG(FATAL) << "Store statement not supported in OpenGL."
             << " Texture store should be a Call statement.";
}

// texelFetch(tex, ivec2(idx & kTextureRowMask, idx >> kTextureRowBits), 0).r
std::string CodeGenOpenGL::TexelFetch(const Variable* buffer, Expr index) {
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
    Type t, const Variable* buffer, Expr index) {
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

void CodeGenOpenGL::PrintType(Type t, std::ostream& os) {
  switch (t.code()) {
    case halideir_type_int:
      CHECK_EQ(t.bits(), 32) << "Only support 32-bit int.";
      os << "int";
      break;
    case halideir_type_uint:
      CHECK_EQ(t.bits(), 32) << "Only support 32-bit uint.";
      os << "uint";
      break;
    case halideir_type_float:
      CHECK_EQ(t.bits(), 32) << "Only support 32-bit float.";
      os << "float";
      break;
    default:
      LOG(FATAL) << "Unsupported type code.";
  }
}

// Codegen for immediate values

void CodeGenOpenGL::VisitExpr_(const IntImm* op, std::ostream& os) {
  CHECK_EQ(op->type, Int(32)) << "GLSL 3.0 only supports 32-bit ints.";
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenOpenGL::VisitExpr_(const UIntImm* op, std::ostream& os) {
  CHECK_EQ(op->type, UInt(32)) << "GLSL 3.0 only supports 32-bit uints.";
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenOpenGL::VisitExpr_(const FloatImm* op, std::ostream& os) {
  CHECK_EQ(op->type, Float(32)) << "GLSL 3.0 only supports 32-bit floats.";
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenOpenGL::VisitExpr_(const StringImm*, std::ostream& os) {
  LOG(FATAL) << "GLSL 3.0 doesn't support strings.";
}

void CodeGenOpenGL::VisitStmt_(const Evaluate* op) {
  auto call = op->value.as<Call>();
  if (call == nullptr || call->name != Call::glsl_texture_store) {
    // Fallback to normal logic.
    CodeGenC::VisitStmt_(op);
  }

  CHECK_EQ(call->args.size(), 2);
  auto buffer = call->args[0].as<Variable>();
  auto value = call->args[1];

  // Doesn't support store to vector.
  auto type = value.type();
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

}  // namespace codegen
}  // namespace tvm
