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
#include "codegen_metal.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../../runtime/metal/metal_module.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"

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

CodeGenMetal::CodeGenMetal(Target target) : target_(target) {
  decl_stream << "#include <metal_stdlib>\n";
  decl_stream << "using namespace metal;\n\n";
  decl_stream << "union __TVMArgUnion {\n"
              << " int v_int[2];\n"
              << "};\n\n";
}

void CodeGenMetal::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // skip the first underscore, so SSA variable starts from _1
  name_supply_->FreshName("_");

  // add to alloc buffer type.
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  // Function header.
  this->stream << "kernel void " << static_cast<std::string>(global_symbol.value()) << "(";

  // Buffer arguments
  size_t num_buffer = 0;
  int limit = target_->GetAttr<Integer>("max_function_args").value().IntValue();
  if (static_cast<int>(f->params.size()) > limit) {
    LOG(WARNING) << "Probably you won't be able to execute your kernel due to high number of "
                    "buffers in the kernel";
  }
  for (size_t i = 0; i < f->params.size(); ++i, ++num_buffer) {
    Var v = f->params[i];
    if (!v.dtype().is_handle()) break;
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
    stream << ' ' << vid << " [[ buffer(" << i << ") ]],\n";
  }
  // Setup normal arguments.
  size_t nargs = f->params.size() - num_buffer;
  std::string varg = name_supply_->FreshName("arg");
  if (nargs != 0) {
    std::string arg_buf_type = static_cast<std::string>(global_symbol.value()) + "_args_t";
    stream << "  constant " << arg_buf_type << "& " << varg << " [[ buffer(" << num_buffer
           << ") ]],\n";
    // declare the struct
    decl_stream << "struct " << arg_buf_type << " {\n";
    for (size_t i = num_buffer; i < f->params.size(); ++i) {
      Var v = f->params[i];
      ICHECK(!v.dtype().is_handle());
      std::string vid = AllocVarID(v.get());
      std::ostringstream vref;
      if (v.dtype().bits() == 32) {
        decl_stream << "  ";
        PrintType(v.dtype(), decl_stream);
        decl_stream << " " << vid << "[2];\n";
        vref << varg << "." << vid << "[0]";
      } else if (v.dtype().bits() == 64) {
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
  ICHECK_EQ(name_supply_->FreshName("threadIdx"), "threadIdx");
  ICHECK_EQ(name_supply_->FreshName("blockIdx"), "blockIdx");
  int work_dim = 0;
  auto thread_axis = f->GetAttr<Array<tir::IterVar>>(tir::attr::kDeviceThreadAxis).value();

  for (IterVar iv : thread_axis) {
    runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
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
    ICHECK(!var_idmap_.count(iv->var.get()));
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
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, DataType::UInt(thread_index_bits_), iv->var.dtype());
}

void CodeGenMetal::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    // Need to care about sizes and alignment of half3/float3 because tir representation might not
    // be aware of Metal half3/float3 details and can treat them as just three elements,
    // while sizes and alignmnents of half3/float3 are one element more (half3-8 bytes/
    // float13 - 16bytes).
    // Example of problematic pattern: filling of threadgroup packed array using float3 elements
    // by threads concurrently can lead to datarace and wrong data in threadgroup shared array.
    // packed_(half3/float3) are exactly datatypes dealing with 3 elements and per-element
    // alignment
    if (lanes == 3) {
      os << "packed_";
    }
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    if (t.bits() == 8 && t.lanes() == 4) {
      // directly 4 8 bit int in integer.
      os << "int";
      return;
    }
    switch (t.bits()) {
      case 8:
        os << "char";
        break;
      case 16:
        os << "short";
        break;
      case 32:
        os << "int";
        break;
      case 64:
        os << "long";
        break;
      case 1:
        os << "bool";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
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

void CodeGenMetal::PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                    std::ostream& os) {  // NOLINT(*)
  os << vec << "[" << i << "]";
}

void CodeGenMetal::PrintVecElemStore(const std::string& vec, DataType t, int i,
                                     const std::string& value) {
  this->PrintIndent();
  stream << vec << "[" << i << "]"
         << " = " << value << ";\n";
}

void CodeGenMetal::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  if (scope == "global") {
    os << "device ";
  } else if (scope == "shared") {
    os << "threadgroup ";
  } else {
    os << "thread ";
  }
}

void CodeGenMetal::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
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
  if (op->op.same_as(builtin::reinterpret())) {
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

void CodeGenMetal::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  std::ostringstream temp;
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      temp << "-";
    }
    temp << "INFINITY";
  } else if (std::isnan(op->value)) {
    temp << "NAN";
  } else {
    temp << std::scientific << op->value;
    if (op->dtype.bits() == 32)
      temp << 'f';
    else if (op->dtype.bits() == 16)
      temp << 'h';
  }
  MarkConst(temp.str());
  os << temp.str();
}

runtime::Module BuildMetal(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;

  std::stringstream code;
  std::stringstream source;
  std::string fmt = "metal";
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenMetal: Can only take PrimFunc";
    code << "// Function: " << kv.first->name_hint << std::endl;
    CodeGenMetal cg(target);
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenMetal: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    if (const auto* f = Registry::Get("tvm_callback_metal_compile")) {
      source << fsource;
      fsource = (*f)(fsource).operator std::string();
      fmt = "metallib";
    }
    code << fsource;
  }

  return MetalModuleCreate(code.str(), fmt, ExtractFuncInfo(mod), source.str());
}

TVM_REGISTER_GLOBAL("target.build.metal").set_body_typed(BuildMetal);
}  // namespace codegen
}  // namespace tvm
