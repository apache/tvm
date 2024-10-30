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

#include <tvm/tir/transform.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

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

void CodeGenMetal::AddFunction(const GlobalVar& gvar, const PrimFunc& func) {
  // NOTE: There is no inter-function calls among Metal kernels.
  // For now we keep the metal codegen without inter-function call
  // process.
  // We can switch to follow the flow with inter-function call process
  // after the Metal function declaration is properly printed.
  // In Metal, for PrimFuncs with signature
  //    def func(A: Buffer, B: Buffer, x: int, y: float) -> None
  // where there are trailing pod parameters, the codegen emits a struct
  //    struct func_params{ x: int; y: float; }
  // for the function. In the flow of inter-function call process,
  // the struct will be emitted for every time a function is declared.
  // So consequently there are duplicate appearances of a same struct,
  // which makes the Metal compiler unable to recognize.

  // clear previous generated state.
  this->InitFuncState(func);
  // skip the first underscore, so SSA variable starts from _1
  name_supply_->FreshName("v_");

  // add to alloc buffer type.
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  // Function header.
  this->stream << "kernel void " << static_cast<std::string>(global_symbol.value()) << "(";

  // Buffer arguments
  size_t num_buffer = 0;
  size_t limit = target_->GetAttr<Integer>("max_function_args").value().IntValue();
  if (func->params.size() > limit) {
    LOG(WARNING) << "Probably you won't be able to execute your kernel due to high number of "
                    "buffers in the kernel";
  }
  for (size_t i = 0; i < func->params.size(); ++i, ++num_buffer) {
    Var v = func->params[i];
    if (!v.dtype().is_handle()) break;
    this->stream << "  ";
    std::string vid = AllocVarID(v.get());
    auto it = alloc_storage_scope_.find(v.get());
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, this->stream);
    }
    PrintType(GetType(v), this->stream);
    // Register handle data type
    // TODO(tvm-team): consider simply keep type info in the
    // type annotation(via a normalizing rewriting).
    if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(v.get(), prim->dtype);
      }
    }
    this->stream << ' ' << vid << " [[ buffer(" << i << ") ]],\n";
  }
  // Setup normal arguments.
  size_t nargs = func->params.size() - num_buffer;
  std::string varg = name_supply_->FreshName("arg");
  if (nargs != 0) {
    std::string arg_buf_type = static_cast<std::string>(global_symbol.value()) + "_args_t";
    this->stream << "  constant " << arg_buf_type << "& " << varg << " [[ buffer(" << num_buffer
                 << ") ]],\n";
    // declare the struct
    decl_stream << "struct " << arg_buf_type << " {\n";
    for (size_t i = num_buffer; i < func->params.size(); ++i) {
      Var v = func->params[i];
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
  auto launch_params = func->GetAttr<Array<String>>(tir::attr::kKernelLaunchParams).value();
  for (const auto& tag : launch_params) {
    if (tag != runtime::launch_param::kUseDynamicSharedMemoryTag) {
      runtime::ThreadScope scope = runtime::ThreadScope::Create(tag);
      work_dim = std::max(work_dim, scope.dim_index + 1);
    }
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
  thread_work_dim_ = work_dim;

  // the function scope.
  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(func->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenMetal::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  // if we only have threadIdx.x
  // metal will directly print as threadIdx
  std::string vname = iv->thread_tag;
  if (thread_work_dim_ <= 1) {
    vname = vname.substr(0, iv->thread_tag.length() - 2);
  }
  var_idmap_[iv->var.get()] =
      CastFromTo(vname, DataType::UInt(thread_index_bits_), iv->var.dtype());
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
  } else if (t.is_bfloat16()) {
    os << "bfloat";
    return;
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
  } else if (scope == "local") {
    os << "thread ";
  } else {
    LOG(FATAL) << "Unknown storage scope `" << scope << "`";
  }
}

void CodeGenMetal::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;
  if (scope == "metal.simdgroup") {
    ICHECK(op->dtype == DataType::Float(16) || op->dtype == DataType::Float(32) ||
           op->dtype == DataType::BFloat(16))
        << "Only float16, float32, and bfloat16 are supported, but got " << op->dtype;
    ICHECK(constant_size % 64 == 0)
        << "Only 8x8 matrix is supported, but got " << constant_size << " bytes\n";

    std::ostringstream dtype_os;
    PrintType(op->dtype, dtype_os);
    std::string dtype_str = dtype_os.str();
    simdgroup_dtype_[op->buffer_var.get()] = dtype_str;
    stream << "simdgroup_" << dtype_str << "8x8 " << vid << '[' << constant_size / 64 << "];\n";
  } else {
    PrintStorageScope(scope, stream);
    PrintType(op->dtype, stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenMetal::VisitExpr_(const SelectNode* op, std::ostream& os) {  // NOLINT(*)
  os << "select(" << PrintExpr(op->false_value) << ", " << PrintExpr(op->true_value) << ", "
     << PrintExpr(op->condition) << ")";
}

void CodeGenMetal::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExpr(op->value);
  int lanes = op->dtype.lanes();
  PrintType(op->dtype, os);
  os << "(";
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenMetal::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  CHECK(!op->op.as<GlobalVarNode>())
      << "CodegenMetal does not support inter-function calls, "
      << "but expression " << GetRef<Call>(op) << " calls PrimFunc " << op->op;
  auto f_check_simdgroup_shape = [](PrimExpr col, PrimExpr row) {
    ICHECK(col->IsInstance<IntImmNode>() && row->IsInstance<IntImmNode>())
        << "Only constant shape is supported for simdgroup matrix, but got " << col << "x" << row;
    int col_val = col.as<IntImmNode>()->value;
    int row_val = row.as<IntImmNode>()->value;
    ICHECK(col_val == 8 && row_val == 8)
        << "Only 8x8 matrix is supported, but got " << col_val << "x" << row_val;
  };
  if (op->op.same_as(builtin::make_filled_simdgroup_matrix())) {
    ICHECK_EQ(op->args.size(), 5);
    Var var = runtime::Downcast<Var>(op->args[0]);
    // Get the data type of the simdgroup matrix
    auto it = simdgroup_dtype_.find(var.get());
    ICHECK(it != simdgroup_dtype_.end())
        << "Cannot find variable allocation for simdgroup: " << var;
    const std::string& dtype_str = it->second;
    f_check_simdgroup_shape(op->args[3], op->args[4]);
    os << PrintExpr(var) << "[" << PrintExpr(op->args[1]) << "] = make_filled_simdgroup_matrix<"
       << dtype_str << ", " << PrintExpr(op->args[3]) << ", " << PrintExpr(op->args[4]) << ">("
       << PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_load())) {
    ICHECK_EQ(op->args.size(), 7);
    f_check_simdgroup_shape(op->args[4], op->args[5]);
    os << "simdgroup_load(" << PrintExpr(op->args[0]) << "[" << PrintExpr(op->args[1]) << "], "
       << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3]) << ", 0, "
       << PrintExpr(op->args[6]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_store())) {
    ICHECK_EQ(op->args.size(), 7);
    f_check_simdgroup_shape(op->args[4], op->args[5]);
    os << "simdgroup_store(" << PrintExpr(op->args[0]) << "[" << PrintExpr(op->args[1]) << "], "
       << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3]) << ", 0, "
       << PrintExpr(op->args[6]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_multiply_accumulate())) {
    ICHECK_EQ(op->args.size(), 8);
    os << "simdgroup_multiply_accumulate("                                  //
       << PrintExpr(op->args[0]) << "[" << PrintExpr(op->args[1]) << "], "  //
       << PrintExpr(op->args[2]) << "[" << PrintExpr(op->args[3]) << "], "  //
       << PrintExpr(op->args[4]) << "[" << PrintExpr(op->args[5]) << "], "  //
       << PrintExpr(op->args[6]) << "[" << PrintExpr(op->args[7]) << "])";
  } else if (op->op.same_as(builtin::reinterpret())) {
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
  mod = tir::transform::PointerValueTypeRewrite()(std::move(mod));

  std::ostringstream source_maker;
  std::unordered_map<std::string, std::string> smap;
  const auto* fmetal_compile = Registry::Get("tvm_callback_metal_compile");
  std::string fmt = fmetal_compile ? "metallib" : "metal";

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenMetal: Can only take PrimFunc";
    auto global_symbol = kv.second->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.defined());
    std::string func_name = global_symbol.value();

    source_maker << "// Function: " << func_name << "\n";
    CodeGenMetal cg(target);
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenMetal: expect calling_conv equals CallingConv::kDeviceKernelLaunch";

    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    source_maker << fsource << "\n";
    if (fmetal_compile) {
      fsource = (*fmetal_compile)(fsource, target).operator std::string();
    }
    smap[func_name] = fsource;
  }

  return MetalModuleCreate(smap, ExtractFuncInfo(mod), fmt, source_maker.str());
}

TVM_REGISTER_GLOBAL("target.build.metal").set_body_typed(BuildMetal);
}  // namespace codegen
}  // namespace tvm
