/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_opengl.cc
 */
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "./codegen_opengl.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace codegen {

CodeGenOpenGL::CodeGenOpenGL() {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "CodeGenOpenGL::CodeGenOpenGL";
}

void CodeGenOpenGL::AddFunction(LoweredFunc f) {
  LOG_INFO.stream() << "CodeGenOpenGL::AddFunction";

  this->stream << "#version 330 core\n";

  // clear previous generated state.
  this->InitFuncState(f);
  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");
  // add to alloc buffer type.
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  // Allocate argument names.
  for (Var arg : f->args) {
    LOG_INFO.stream() << "Arg: " << arg.get()->name_hint;
    AllocVarID(arg.get());
  }

  // Declare arguments in shader.
  for (size_t i = 1; i < f->args.size(); ++i) {
    Var arg = f->args[i];
    this->stream << "uniform sampler2D " << this->GetVarID(arg.get()) << ";\n";
  }

  CHECK(!f->args.empty()) << "Must have at least one argument";
  this->stream << "out vec4 " << this->GetVarID(f->args[0].get()) << ";\n";
  this->output_ = f->args[0].get();

  this->stream << "void main() {\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenOpenGL::BindThreadIndex(const IterVar &iv) {
  LOG_INFO.stream() << "CodeGenOpenGL::BindThreadIndex";
  CHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] = iv->thread_tag;
}

void CodeGenOpenGL::VisitStmt_(const Store *op) {
  LOG_INFO.stream() << "CodeGenOpenGL::VisitStmt_(const Store *)";
  Type t = op->value.type();
  if (t.lanes() == 1) {
    // Store to a scalar.

    std::string value = this->PrintExpr(op->value);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();
    stream << ref << " = " << value << ";\n";

  } else {
    // Store to an array.
    LOG_FATAL.stream() << "Storing to an array not implemented";
  }
}

// Print a reference expression to a buffer.
// buffer[index] => texelFetch(buffer, index, 0).r
std::string CodeGenOpenGL::GetBufferRef(
    Type t, const Variable* buffer, Expr index) {
  if (buffer == this->output_) {
    return this->var_idmap_[buffer] + ".r";
  }

  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  CHECK(t.lanes() == 1) << "Vector type not supported";
  CHECK(HandleTypeMatch(buffer, t)) << "Type mismatch not supported";

  os << "texelFetch(" << vid << ", ";
  PrintExpr(index, os);
  os << ", 0).r";

  return os.str();
}

}  // namespace tvm
}  // namespace codegen
