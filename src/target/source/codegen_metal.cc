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
 * \file codegen_metal.cc
 */
#include <vector>
#include <string>
#include <algorithm>
#include "codegen_metal.h"
#include "../build_common.h"
#include "../../runtime/metal/metal_module.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace codegen {

void CodeGenMetal::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  // analyze the data;
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

CodeGenMetal::CodeGenMetal() {
  decl_stream << "#include <metal_stdlib>\n";
  decl_stream << "using namespace metal;\n\n";
  decl_stream << "union __TVMArgUnion {\n"
              << " int v_int;\n"
              << "};\n\n";
}

void CodeGenMetal::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");

  // add to alloc buffer type.
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  CHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  // Function header.
  this->stream << "kernel void " << static_cast<std::string>(global_symbol.value()) << "(";

  // Buffer arguments
  size_t num_buffer = 0;
  for (size_t i = 0; i < f->params.size(); ++i, ++num_buffer) {
    Var v = f->params[i];
    if (!v.dtype().is_handle())  break;
    stream << "  ";
    std::string vid = AllocVarID(v.get());
    auto it = alloc_storage_scope_.find(v.get());
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, stream);
    }
    PrintType(GetType(v), stream);
    // Register handle data type
    // TODO(tvm-team): consider simply keep type info in the
    // type annotation(via a normalizing rewriting).
    if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(v.get(), prim->dtype);
      }
    }
    stream << ' ' << vid
           << " [[ buffer(" << i << ") ]],\n";
  }
  // Setup normal arguments.
  size_t nargs = f->params.size() - num_buffer;
  std::string varg = GetUniqueName("arg");
  if (nargs != 0) {
    std::string arg_buf_type =
        static_cast<std::string>(global_symbol.value()) + "_args_t";
    stream << "  constant " << arg_buf_type << "& " << varg
           << " [[ buffer(" << num_buffer << ") ]],\n";
    // declare the struct
    decl_stream << "struct " << arg_buf_type << " {\n";
    for (size_t i = num_buffer; i < f->params.size(); ++i) {
      Var v = f->params[i];
      CHECK(!v.dtype().is_handle());
      std::string vid = AllocVarID(v.get());
      std::ostringstream vref;
      if (v.dtype().bits() == 32) {
        decl_stream << "  ";
        PrintType(v.dtype(), decl_stream);
        decl_stream << " " << vid << ";\n";
        vref << varg << "." << vid;
      } else {
        // For non 32bit type, ref through arg union.
        decl_stream << "  __TVMArgUnion " << vid << ";\n";
        vref << varg << "." << vid << ".v_";
        PrintType(v.dtype(), vref);
      }
      var_idmap_[v.get()] = vref.str();
    }
    decl_stream << "};\n\n";
  }
  // Setup the thread group info.
  CHECK_EQ(GetUniqueName("threadIdx"), "threadIdx");
  CHECK_EQ(GetUniqueName("blockIdx"), "blockIdx");
  int work_dim = 0;
  auto thread_axis = f->GetAttr<Array<tir::IterVar>>(
      tir::attr::kDeviceThreadAxis).value();

  for (IterVar iv : thread_axis) {
    runtime::ThreadScope scope = runtime::ThreadScope::make(iv->thread_tag);
    work_dim = std::max(work_dim, scope.dim_index + 1);
  }
  if (work_dim != 0) {
    // use ushort by default for now
    stream << "  ";
    PrintType(DataType::UInt(thread_index_bits_, work_dim), stream);
    stream << " blockIdx [[threadgroup_position_in_grid]],\n";
    stream << "  ";
    PrintType(DataType::UInt(thread_index_bits_, work_dim), stream);
    stream << " threadIdx [[thread_position_in_threadgroup]]\n";
  }
  // bind thread axis
  for (IterVar iv : thread_axis) {
    CHECK(!var_idmap_.count(iv->var.get()));
    std::string vname = iv->thread_tag;
    if (work_dim <= 1) {
      vname = vname.substr(0, iv->thread_tag.length() - 2);
    }
    var_idmap_[iv->var.get()] =
        CastFromTo(vname, DataType::UInt(thread_index_bits_), iv->var.dtype());
  }
  // the function scope.
  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenMetal::BindThreadIndex(const IterVar& iv) {
  CHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, DataType::UInt(thread_index_bits_), iv->var.dtype());
}

void CodeGenMetal::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1)
        << "do not yet support vector types";
    os << "void*"; return;
  }
  if (t == DataType::Bool()) {
    os << "bool"; return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16: os << "half"; break;
      case 32: os << "float"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    if (t.bits() == 8 && t.lanes() == 4) {
      // directly 4 8 bit int in integer.
      os << "int"; return;
    }
    switch (t.bits()) {
      case 8: os << "char"; break;
      case 16: os << "short"; break;
      case 32: os << "int"; break;
      case 1: os << "bool"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Metal type";
}

void CodeGenMetal::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    this->PrintIndent();
    this->stream << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "global") {
    LOG(FATAL) << "global barrier not supported";
  }
}

void CodeGenMetal::PrintVecElemLoad(const std::string& vec,
                                    DataType t, int i,
                                    std::ostream& os) {  // NOLINT(*)
  os << vec << "[" << i << "]";
}

void CodeGenMetal::PrintVecElemStore(const std::string& vec,
                                     DataType t, int i,
                                     const std::string& value) {
  this->PrintIndent();
  stream << vec << "[" << i << "]"
         << " = " << value << ";\n";
}

void CodeGenMetal::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
  if (scope == "global") {
    os << "device ";
  } else if (scope == "shared") {
    os << "threadgroup ";
  } else {
    os << "thread ";
  }
}

void CodeGenMetal::VisitExpr_(const BroadcastNode* op, std::ostream& os) {   // NOLINT(*)
  std::string v = PrintExpr(op->value);
  PrintType(op->dtype, os);
  os << "(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenMetal::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->is_intrinsic(CallNode::reinterpret)) {
    // generate as_type<TYPE>(ARG)
    os << "(as_type<";
    this->PrintType(op->dtype, os);
    os << ">(";
    this->PrintExpr(op->args[0], os);
    os << "))";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

runtime::Module BuildMetal(IRModule mod) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenMetal cg;
  cg.Init(output_ssa);

  for (auto kv :  mod->functions) {
    CHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenMetal: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    CHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenMetal: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();
  std::string fmt = "metal";
  std::string source = "";
  if (const auto* f = Registry::Get("tvm_callback_metal_compile")) {
    source = code;
    code = (*f)(code).operator std::string();
    fmt = "metallib";
  }
  return MetalModuleCreate(code, fmt, ExtractFuncInfo(mod), source);
}

TVM_REGISTER_GLOBAL("target.build.metal")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildMetal(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
