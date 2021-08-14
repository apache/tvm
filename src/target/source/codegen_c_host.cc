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
#include "codegen_c_host.h"

#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>

#include <string>
#include <vector>

#include "../../support/str_escape.h"
#include "../build_common.h"
#include "../func_registry_generator.h"
#include "codegen_params.h"

namespace tvm {
namespace codegen {

CodeGenCHost::CodeGenCHost() { module_name_ = GetUniqueName("__tvm_module_ctx"); }

void CodeGenCHost::Init(bool output_ssa, bool emit_asserts, std::string target_str) {
  emit_asserts_ = emit_asserts;
  declared_globals_.clear();
  decl_stream << "// tvm target: " << target_str << "\n";
  decl_stream << "#define TVM_EXPORTS\n";
  decl_stream << "#include \"tvm/runtime/c_runtime_api.h\"\n";
  decl_stream << "#include \"tvm/runtime/c_backend_api.h\"\n";
  decl_stream << "#include <math.h>\n";
  CodeGenC::Init(output_ssa);
}

void CodeGenCHost::DefineModuleName() { decl_stream << "void* " << module_name_ << " = NULL;\n"; }

void CodeGenCHost::AddFunction(const PrimFunc& f) {
  auto global_symbol = f->attrs.GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenCHost: Expect PrimFunc to have the global_symbol attribute";
  function_names_.push_back(global_symbol.value());

  CodeGenC::AddFunction(f);
}

void CodeGenCHost::DeclareParameters(Map<String, LinkedParam> params) {
  for (auto kv : params) {
    decl_stream << "\n"
                << "#ifdef __cplusplus\n"
                << "extern \"C\" {\n"
                << "#endif\n"
                << "static const ";
    int64_t num_elements = 1;
    for (int64_t dim : kv.second->param.Shape()) {
      num_elements *= dim;
    }
    PrintType(kv.second->param.DataType(), decl_stream);
    decl_stream << " " << ::tvm::runtime::symbol::tvm_param_prefix << kv.first << "["
                << num_elements << "] = {\n";
    NDArrayDataToC(kv.second->param, 4, decl_stream);
    decl_stream << "};\n"
                << "#ifdef __cplusplus\n"
                << "}  // extern \"C\"\n"
                << "#endif\n";
  }
}

void CodeGenCHost::LinkParameters(Map<String, LinkedParam> params) {
  PrintFuncPrefix();
  stream << " " << tvm::runtime::symbol::tvm_lookup_linked_param
         << "(void* args, int* arg_type_ids, int num_args, void* out_ret_value, "
         << "int* out_ret_tcode, void* resource_handle) {\n";
  ICHECK_EQ(GetUniqueName(tvm::runtime::symbol::tvm_lookup_linked_param),
            tvm::runtime::symbol::tvm_lookup_linked_param)
      << "builtin PackedFunc name already taken: " << tvm::runtime::symbol::tvm_lookup_linked_param;
  stream << "    switch (((int64_t*) args)[0]) {\n"
         << "    default:\n"
         << "        out_ret_tcode[0] = " << kTVMNullptr << ";\n"
         << "        return 0;\n";

  function_names_.push_back(tvm::runtime::symbol::tvm_lookup_linked_param);
  for (auto kv : params) {
    stream << "    case " << kv.second->id << ":\n"
           << "        ((uint64_t*)out_ret_value)[0] = (uint64_t) (uintptr_t) "
           << ::tvm::runtime::symbol::tvm_param_prefix << kv.first << ";\n"
           << "        out_ret_tcode[0] = " << kTVMOpaqueHandle << ";\n"
           << "        return 0;\n";
  }
  stream << "    }\n"
         << "}\n";
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
    ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8:
        os << "int8_t";
        break;
      case 16:
        os << "int16_t";
        break;
      case 32:
        os << "int32_t";
        break;
      case 64:
        os << "int64_t";
        break;
      case 1:
        os << "int32_t";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

void CodeGenCHost::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
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
  this->stream << "if (TVMBackendGetFuncFromEnv(" << module_name_ << ", \"" << func_name << "\""
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
               << "(TVMValue*) stack_value"
               << ", "
               << "(int*) stack_tcode"
               << ", " << num_args << ", "
               << "&" << ret_val << ", "
               << "&" << ret_type_code << ") != 0) {\n";
  int func_call_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(func_call_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

void CodeGenCHost::PrintFuncCallC(const std::string& packed_func_name, int num_args) {
  this->PrintIndent();
  std::string ret_val = GetUniqueName("ret_val");
  std::string ret_type_code = GetUniqueName("ret_type_code");
  this->stream << "TVMValue " << ret_val << ";\n";
  this->PrintIndent();
  this->stream << "int " << ret_type_code << ";\n";
  this->PrintIndent();

  this->stream << "if (" << packed_func_name << "( "
               << "(TVMValue*) stack_value "
               << ", "
               << "(int*) stack_tcode"
               << ", " << num_args << ", "
               << "&" << ret_val << ", "
               << "&" << ret_type_code << ", NULL) != 0){\n";

  int func_call_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "return -1;\n";
  this->EndScope(func_call_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

CodeGenCHost::FunctionInfo CodeGenCHost::GetFunctionInfo(const CallNode* op) {
  const StringImmNode* s = op->args[0].as<StringImmNode>();
  ICHECK(s != nullptr) << "tvm_call_packed_lowered expects first argument as function name";
  int64_t begin = op->args[3].as<IntImmNode>()->value;
  int64_t end = op->args[4].as<IntImmNode>()->value;
  int64_t num_args = end - begin;
  ICHECK_GE(num_args, 0);
  std::string func_name = s->value;
  // NOTE: cannot rely on GetUnique for global decl_stream declarations
  // because it is reset between AddFunction().
  std::string packed_func_name = func_name + "_packed";
  std::string unique_name;
  auto it = declared_globals_.find(packed_func_name);
  if (it != declared_globals_.end()) {
    unique_name = it->second;
  } else {
    unique_name = GetUniqueName(packed_func_name);
    declared_globals_[packed_func_name] = unique_name;
    decl_stream << "static void* " << unique_name << " = NULL;\n";
  }
  return {func_name, unique_name, num_args};
}

void CodeGenCHost::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->op.same_as(builtin::tvm_stack_alloca())) {
    std::string stack_name = GetUniqueName("stack");
    const std::string& type = op->args[0].as<StringImmNode>()->value;
    const IntImmNode* num = op->args[1].as<IntImmNode>();
    ICHECK(num != nullptr);
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
  } else if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    auto function_info = GetFunctionInfo(op);
    this->PrintGetFuncFromBackend(function_info.func_name, function_info.func_name_packed);
    this->PrintFuncCall(function_info.func_name_packed, function_info.num_args);
  } else if (op->op.same_as(builtin::tvm_call_cpacked_lowered())) {
    auto function_info = GetFunctionInfo(op);
    this->PrintFuncCallC(function_info.func_name, function_info.num_args);
  } else if (op->op.same_as(builtin::tvm_throw_last_error())) {
    this->PrintIndent();
    this->stream << "return -1;\n";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCHost::VisitStmt_(const AssertStmtNode* op) {  // NOLINT(*)
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

void CodeGenCHost::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, "<", os);
}

void CodeGenCHost::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  PrintTernaryCondExpr(op, ">", os);
}

template <typename T>
inline void CodeGenCHost::PrintTernaryCondExpr(const T* op, const char* compare,
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

runtime::Module BuildCHost(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  bool emit_asserts = false;
  CodeGenCHost cg;
  cg.Init(output_ssa, emit_asserts, target->str());

  Map<String, LinkedParam> linked_params;
  bool found_linked_params = false;
  bool could_have_linked_params = target->GetAttr<Bool>("link-params").value_or(Bool(false));
  PrimFunc aot_executor_fn;

  for (auto kv : mod->functions) {
    if (could_have_linked_params &&
        kv.first->name_hint == ::tvm::runtime::symbol::tvm_lookup_linked_param) {
      Map<String, ObjectRef> attrs_dict = Downcast<Map<String, ObjectRef>>(kv.second->attrs->dict);
      CHECK(attrs_dict.find(::tvm::tir::attr::kLinkedParams) != attrs_dict.end())
          << "no " << ::tvm::tir::attr::kLinkedParams << " attribute found!";
      linked_params =
          Downcast<Map<String, LinkedParam>>(attrs_dict[::tvm::tir::attr::kLinkedParams]);
      found_linked_params = true;
      continue;
    }
    // Make sure that the executor function is the last one to be code generated so that all the
    // symbols are available to tvm_run_func
    auto fun_name = std::string(kv.first->name_hint);
    bool is_aot_executor_fn =
        kv.second->attrs.GetAttr<Bool>("runner_function", Bool(false)).value();

    if (is_aot_executor_fn) {
      aot_executor_fn = Downcast<PrimFunc>(kv.second);
      continue;
    }

    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodegenCHost: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(f);
  }

  if (could_have_linked_params && !aot_executor_fn.defined()) {
    ICHECK(found_linked_params) << "-link-params given but none found";
    cg.DeclareParameters(linked_params);
    cg.LinkParameters(linked_params);
  }

  if (could_have_linked_params && aot_executor_fn.defined()) {
    cg.DeclareParameters(linked_params);
    cg.AddFunction(aot_executor_fn);
  }

  if (target->GetAttr<Bool>("system-lib").value_or(Bool(false))) {
    ICHECK_EQ(target->GetAttr<String>("runtime").value_or(""), "c")
        << "c target only supports generating C runtime SystemLibs";
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_REGISTER_GLOBAL("target.build.c").set_body_typed(BuildCHost);
}  // namespace codegen
}  // namespace tvm
