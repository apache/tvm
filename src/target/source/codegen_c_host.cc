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
 * \file codegen_c_host.cc
 */
#include <tvm/target/codegen.h>
#include <vector>
#include <string>
#include "../build_common.h"
#include "codegen_c_host.h"

namespace tvm {
namespace codegen {

CodeGenCHost::CodeGenCHost() {
  module_name_ = GetUniqueName("__tvm_module_ctx");
}

void CodeGenCHost::Init(bool output_ssa, bool emit_asserts) {
  emit_asserts_ = emit_asserts;
  declared_globals_.clear();
  decl_stream << "#include \"tvm/runtime/c_runtime_api.h\"\n";
  decl_stream << "#include \"tvm/runtime/c_backend_api.h\"\n";
  decl_stream << "void* " << module_name_ << " = NULL;\n";
  CodeGenC::Init(output_ssa);
}

void CodeGenCHost::PrintFuncPrefix() {  // NOLINT(*)
  stream << "#ifdef __cplusplus\n"
         << "extern \"C\"\n"
         << "#endif\n"
         << "TVM_DLL int32_t";
}

void CodeGenCHost::PrintFinalReturn() {  // NOLINT(*)
  this->PrintIndent();
  stream << "return 0;\n";
}

void CodeGenCHost::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1)
        << "does not support vector types";
    os << "void*"; return;
  }
  if (t == DataType::Bool()) {
    os << "bool"; return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32: os << "float"; break;
      case 64:
        os << "double";
        break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes; return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8: os << "int8_t"; break;
      case 16: os << "int16_t"; break;
      case 32: os << "int32_t"; break;
      case 64: os << "int64_t"; break;
      case 1: os << "int32_t"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

void CodeGenCHost::VisitExpr_(const BroadcastNode* op, std::ostream& os) {   // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "((";
  PrintType(op->dtype, os);
  os << ")(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "))";
}

void CodeGenCHost::PrintGetFuncFromBackend(const std::string& func_name,
                                           const std::string& packed_func_name) {
  this->PrintIndent();
  this->stream << "if (" << packed_func_name << " == NULL) {\n";
  int packed_func_if_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "if (TVMBackendGetFuncFromEnv(" << module_name_
              << ", \"" << func_name << "\""
              << ", &" << packed_func_name << ") != 0) {\n";
  int get_func_env_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(get_func_env_scope);
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(packed_func_if_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenCHost::PrintFuncCall(const std::string& packed_func_name, int num_args) {
  this->PrintIndent();
  std::string ret_val = GetUniqueName("ret_val");
  std::string ret_type_code = GetUniqueName("ret_type_code");
  this->stream << "TVMValue " << ret_val << ";\n";
  this->PrintIndent();
  this->stream << "int " << ret_type_code << ";\n";
  this->PrintIndent();
  this->stream << "if (TVMFuncCall(" << packed_func_name << ", "
               << "(TVMValue*) stack_value" << ", " << "(int*) stack_tcode" << ", "
               << num_args << ", " << "&" << ret_val << ", " << "&"
               << ret_type_code << ") != 0) {\n";
  int func_call_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(func_call_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenCHost::VisitExpr_(const CallNode *op, std::ostream& os) { // NOLINT(*)
  if (op->is_intrinsic(intrinsic::tvm_stack_alloca)) {
    std::string stack_name = GetUniqueName("stack");
    const std::string& type = op->args[0].as<StringImmNode>()->value;
    const IntImmNode* num = op->args[1].as<IntImmNode>();
    CHECK(num != nullptr);
    static_assert(alignof(TVMValue) % alignof(DLTensor) == 0, "invariant");
    size_t unit = sizeof(TVMValue);
    size_t size = 0;
    if (type == "shape") {
      size = (num->value * sizeof(tvm_index_t) + unit - 1) / unit;
    } else if (type == "arg_value") {
      size = (num->value * sizeof(TVMValue) + unit - 1) / unit;
    } else if (type == "arg_tcode") {
      size = (num->value * sizeof(int) + unit - 1) / unit;
    } else if (type == "array") {
      size = (num->value * sizeof(DLTensor) + unit - 1) / unit;
    } else {
      LOG(FATAL) << "Unknown stack alloca type " << type;
    }
    this->PrintIndent();
    this->stream << "TVMValue " << stack_name << "[" << size << "];\n";
    os << stack_name;
  } else if (op->is_intrinsic(intrinsic::tvm_call_packed_lowered)) {
    const StringImmNode* s = op->args[0].as<StringImmNode>();
    CHECK(s != nullptr) << "tvm_call_packed_lowered expects first argument as function name";
    int64_t begin = op->args[3].as<IntImmNode>()->value;
    int64_t end = op->args[4].as<IntImmNode>()->value;
    int64_t num_args = end - begin;
    CHECK_GE(num_args, 0);
    std::string func_name = s->value;
    // NOTE: cannot rely on GetUnique for global decl_stream declarations
    // because it is reset between AddFunction().
    std::string packed_func_name = func_name + "_packed";
    if (declared_globals_.insert(packed_func_name).second) {
      // Still reserve the name among unique names.
      CHECK(GetUniqueName(packed_func_name) == packed_func_name) <<
        "Expected name " << packed_func_name << " to not be taken";
      decl_stream << "static void* " << packed_func_name << " = NULL;\n";
    }
    this->PrintGetFuncFromBackend(func_name, packed_func_name);
    this->PrintFuncCall(packed_func_name, num_args);
  } else if (op->is_intrinsic(intrinsic::tvm_throw_last_error)) {
    this->PrintIndent();
    this->stream << "return -1;\n";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCHost::VisitStmt_(const AssertStmtNode *op) { // NOLINT(*)
  if (emit_asserts_) {
    std::string cond = PrintExpr(op->condition);
    PrintIndent();
    stream << "if (!(" << cond << ")) {\n";
    int assert_if_scope = this->BeginScope();
    PrintIndent();
    stream << "TVMAPISetLastError(\"" << op->message.as<StringImmNode>()->value << "\");\n";
    PrintIndent();
    stream << "return -1;\n";
    this->EndScope(assert_if_scope);
    PrintIndent();
    stream << "}\n";
  }
  this->PrintStmt(op->body);
}

void CodeGenCHost::VisitExpr_(const MinNode *op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, "<", os);
}

void CodeGenCHost::VisitExpr_(const MaxNode *op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, ">", os);
}

template <typename T>
inline void CodeGenCHost::PrintTernaryCondExpr(const T* op,
                                           const char* compare,
                                           std::ostream& os) {  // NOLINT(*)
  std::ostringstream temp_a;
  VisitExpr(op->a, temp_a);
  std::string a_id = SSAGetID(temp_a.str(), op->a.dtype());
  std::ostringstream temp_b;
  VisitExpr(op->b, temp_b);
  std::string b_id = SSAGetID(temp_b.str(), op->b.dtype());

  os << "((" << a_id << ") " << compare << " (" << b_id << ") "
     << "? (" << a_id << ") : (" << b_id << "))";
}

runtime::Module BuildCHost(IRModule mod) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  bool emit_asserts = false;
  CodeGenCHost cg;
  cg.Init(output_ssa, emit_asserts);

  for (auto kv : mod->functions) {
    CHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodegenCHost: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c");
}

TVM_REGISTER_GLOBAL("target.build.c")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = BuildCHost(args[0]);
});
}  // namespace codegen
}  // namespace tvm
