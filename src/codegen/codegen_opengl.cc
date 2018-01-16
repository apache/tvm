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
    : output_(nullptr), inputs_(), iter_var_(nullptr) {
  // TODO(zhixunt): Implement this.
  LOG(INFO) << "CodeGenOpenGL::CodeGenOpenGL";
}

void CodeGenOpenGL::InitFuncState(LoweredFunc f) {
  CodeGenC::InitFuncState(f);
  output_ = nullptr;
  inputs_.clear();
  iter_var_ = nullptr;
}

void CodeGenOpenGL::AddFunction(LoweredFunc f) {
  LOG(INFO) << "CodeGenOpenGL::AddFunction";

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
    LOG(INFO) << "Allocated argument name: " << arg.get()->name_hint << " => "
              << arg_name;
    var_idmap_[arg.get()] = arg_name;
  }

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
      auto type = type_it->second;

      switch (type.code()) {
        case halideir_type_int:
          this->decl_stream << "uniform isampler2D " << arg_name << ";\n";
          break;
        case halideir_type_uint:
          this->decl_stream << "uniform usampler2D " << arg_name << ";\n";
          break;
        case halideir_type_float:
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
  std::vector<int> arg_kinds;
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

  shaders_[f->name] = runtime::OpenGLShader{
      .source = this->decl_stream.str() + this->stream.str(),
      .arg_names = std::move(arg_names),
      .arg_kinds = std::move(arg_kinds)
  };
}

std::unordered_map<std::string, runtime::OpenGLShader> CodeGenOpenGL::Finish() {
  return shaders_;
}

void CodeGenOpenGL::BindThreadIndex(const IterVar& iv) {
  LOG(INFO) << "CodeGenOpenGL::BindThreadIndex";
  CHECK(!var_idmap_.count(iv->var.get()));

  // TODO(zhixunt): Can we not hardcode the name?
  CHECK(iv->thread_tag == "threadIdx.x") << "Must be threadIdx.x";
  var_idmap_[iv->var.get()] = iv->thread_tag;

  CHECK(iter_var_ == nullptr) << "Only support one iter var";
  iter_var_ = iv->var.get();

  PrintIndent();
  this->stream << "ivec2 threadIdx = ivec2(gl_FragCoord.xy);\n";
}

// GLSL texture store is special. We can only store to one output texture, and
// we must store to the index that matches the current "thread index".
void CodeGenOpenGL::VisitStmt_(const Store* op) {
  LOG(INFO) << "CodeGenOpenGL::VisitStmt_(const Store *)";
  auto t = op->value.type();
  auto buffer = op->buffer_var.get();
  auto index = op->index.get();

  if (t.lanes() == 1) {
    // Store to a scalar.
    CHECK(inputs_.find(buffer) == inputs_.cend())
      << "Texture has been read from. Must not store to it.";
    if (output_ == nullptr) {
      output_ = buffer;  // Record that this texture is the output.
    } else {
      CHECK(output_ != buffer) << "GLSL can only write to 1 texture.";
    }
    CHECK(index == iter_var_) << "GLSL must write to corresponding elem.";

    this->PrintIndent();
    stream << GetVarID(buffer) << " = " << PrintExpr(op->value) << ";\n";

  } else {
    // Store to a vector.
    LOG(FATAL) << "Vectorized store not implemented.";
  }
}

// Print a reference expression to a buffer.
// Format: texelFetch(buffer, index, 0).r
std::string CodeGenOpenGL::GetBufferRef(
    Type t, const Variable* buffer, Expr index) {
  CHECK(buffer != this->output_)
    << "Texture has been stored to. Must not read from it.";
  CHECK(t.lanes() == 1) << "Vector type not supported.";
  CHECK(HandleTypeMatch(buffer, t)) << "Type mismatch not supported.";

  this->inputs_.insert(buffer);  // Record that this texture is an input.

  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  os << "texelFetch(" << vid << ", ";
  os << "ivec2(";
  PrintExpr(index, os);
  os << ", 0)";
  os << ", 0).r";
  return os.str();
}

void CodeGenOpenGL::PrintType(Type t, std::ostream& os) {
  switch (t.code()) {
    case halideir_type_int:
      CHECK(t.bits() == 32) << "Only support 32-bit int.";
      os << "int";
      break;
    case halideir_type_uint:
      CHECK(t.bits() == 32) << "Only support 32-bit uint.";
      os << "uint";
      break;
    case halideir_type_float:
      CHECK(t.bits() == 32) << "Only support 32-bit float.";
      os << "float";
      break;
    default:
      LOG(FATAL) << "Unsupported type code.";
  }
}

// Codegen for immediate values

void CodeGenOpenGL::VisitExpr_(const IntImm* op, std::ostream& os) {
  CHECK(op->type == Int(32)) << "GLSL 3.0 only supports 32-bit ints.";
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenOpenGL::VisitExpr_(const UIntImm* op, std::ostream& os) {
  CHECK(op->type == UInt(32)) << "GLSL 3.0 only supports 32-bit uints.";
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenOpenGL::VisitExpr_(const FloatImm* op, std::ostream& os) {
  CHECK(op->type == Float(32)) << "GLSL 3.0 only supports 32-bit floats.";
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenOpenGL::VisitExpr_(const StringImm*, std::ostream& os) {
  LOG(FATAL) << "GLSL 3.0 doesn't support strings.";
}

}  // namespace codegen
}  // namespace tvm
