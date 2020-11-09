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
 * \file codegen_opencl.cc
 */
#include "codegen_opencl.h"

#include <cmath>
#include <string>
#include <vector>

#include "../../runtime/opencl/opencl_module.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {

CodeGenOpenCL::CodeGenOpenCL() { restrict_keyword_ = "restrict"; }

void CodeGenOpenCL::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

void CodeGenOpenCL::PrintFuncPrefix() { stream << "__kernel void"; }

std::string CodeGenOpenCL::Finish() {
  // inject extension enable pragma for fp16 and fp64
  if (enable_fp16_) {
    decl_stream << "#ifdef cl_khr_fp16\n"
                   "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                   "#elif defined(cl_amd_fp16)\n"
                   "#pragma OPENCL EXTENSION cl_amd_fp16 : enable\n"
                   "#else\n"
                   "#error \"Half precision floating point not supported"
                   "by OpenCL implementation on your device.\" \n"
                   "#endif\n\n";
  }

  if (enable_fp64_) {
    decl_stream << "#ifdef cl_khr_fp64\n"
                   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                   "#elif defined(cl_amd_fp64)\n"
                   "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
                   "#else\n"
                   "#error \"Double precision floating point not supported"
                   "by OpenCL implementation on your device.\" \n"
                   "#endif\n\n";
  }

  // Enable atomic_add used by get_valid_counts. Only needed for OpenCL < 1.1.
  if (enable_atomics_) {
    decl_stream << "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
                   "#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable\n\n";
  }
  return CodeGenC::Finish();
}

void CodeGenOpenCL::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
  std::ostringstream os;
  if (ts.rank == 1) {
    os << "get_local_id(" << ts.dim_index << ")";
  } else {
    os << "get_group_id(" << ts.dim_index << ")";
  }
  var_idmap_[iv->var.get()] = CastFromTo(os.str(), DataType::UInt(64), iv->var.dtype());
}

void CodeGenOpenCL::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "do not yet support vector types";
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
        enable_fp16_ = true;
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        enable_fp64_ = true;
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && ((lanes >= 2 && lanes <= 4) || lanes == 8 || lanes == 16)) {
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
        os << "int";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && ((lanes >= 2 && lanes <= 4) || lanes == 8 || lanes == 16)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to OpenCL type";
}

void CodeGenOpenCL::PrintVecAddr(const VarNode* buffer, DataType t, PrimExpr base,
                                 std::ostream& os) {  // NOLINT(*)
  if (!HandleTypeMatch(buffer, t.element_of())) {
    os << '(';
    auto it = alloc_storage_scope_.find(buffer);
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    PrintType(t.element_of(), os);
    os << "*)";
  }
  os << GetVarID(buffer) << " + ";
  PrintExpr(base, os);
}
std::string CodeGenOpenCL::GetVecLoad(DataType t, const VarNode* buffer, PrimExpr base) {
  std::ostringstream os;
  os << "vload" << t.lanes() << "(0, ";
  PrintVecAddr(buffer, t, base, os);
  os << ")";
  return os.str();
}

void CodeGenOpenCL::PrintVecStore(const VarNode* buffer, DataType t, PrimExpr base,
                                  const std::string& value) {
  this->PrintIndent();
  stream << "vstore" << t.lanes() << "(" << value << ", 0, ";
  PrintVecAddr(buffer, t, base, stream);
  stream << ");\n";
}

void CodeGenOpenCL::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    this->PrintIndent();
    this->stream << "barrier(CLK_LOCAL_MEM_FENCE);\n";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "barrier(CLK_LOCAL_MEM_FENCE);\n";
  } else if (sync == "global") {
    LOG(FATAL) << "not supported";
  }
}

void CodeGenOpenCL::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  if (scope == "global") {
    os << "__global ";
  } else if (scope == "shared") {
    os << "__local ";
  }
}

std::string CodeGenOpenCL::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  std::ostringstream os;
  if (target.lanes() == 1) {
    os << "((";
    this->PrintType(target, os);
    os << ")" << value << ")";
  } else {  // convert vector type
    os << "(";
    os << "convert_";
    this->PrintType(target, os);
    os << "(" << value << "))";
  }
  return os.str();
}

void CodeGenOpenCL::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (op->op.same_as(builtin::address_of())) {
    // Overload tvm_address_of to add storage scope (e.g. __global).
    const LoadNode* load = op->args[0].as<LoadNode>();
    ICHECK(op->args.size() == 1 && load);
    os << "((";
    auto it = alloc_storage_scope_.find(load->buffer_var.get());
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    this->PrintType(load->dtype.element_of(), os);
    os << " *)" << this->GetVarID(load->buffer_var.get()) << " + ";
    this->PrintExpr(load->index, os);
    os << ')';
  } else if (op->op.same_as(builtin::text2d_store())) {
    os << "write_imagef(";
    this->PrintExpr(op->args[0], os);
    os << ", ";
    os << "(int2)(";
    this->PrintExpr(op->args[2], os);
    os << ", ";
    this->PrintExpr(op->args[1], os);
    os << "), ";
    this->PrintExpr(op->args[3], os);
    os << ")";
  } else if (op->op.same_as(builtin::text2d_load())) {
    /*
      float4 read_imagef(read_only image2d_t image,
      sampler_t sampler,
      int2 coord)
    */
    // std::cout << "LOAD\n";
    // std::cout << op->args << std::endl;
    os << "read_imagef(";
    this->PrintExpr(op->args[0], os);
    os << ", ";
    os << "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, ";
    os << "(int2)(";
    this->PrintExpr(op->args[2], os);
    os << ", ";
    this->PrintExpr(op->args[1], os);
    os << "))";
  } else if (op->op.same_as(builtin_call_extern_)) {
    auto func = Downcast<StringImm>(op->args[0]);
    // Enable atomics extension if used.
    if (func->value == "atomic_add") {
      enable_atomics_ = true;
    }
    CodeGenC::VisitExpr_(op, os);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenOpenCL::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
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

void CodeGenOpenCL::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      os << "-";
    }
    os << "INFINITY";
  } else if (std::isnan(op->value)) {
    os << "NAN";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

runtime::Module BuildOpenCL(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;

  std::stringstream code;
  const auto* fpostproc = Registry::Get("tvm_callback_opencl_postproc");
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenOpenCL: Can only take PrimFunc";
    code << "// Function: " << kv.first->name_hint << std::endl;
    CodeGenOpenCL cg;
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenOpenCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    if (fpostproc) {
      fsource = (*fpostproc)(fsource).operator std::string();
    }
    code << fsource;
  }

  return OpenCLModuleCreate(code.str(), "cl", ExtractFuncInfo(mod), code.str());
}

TVM_REGISTER_GLOBAL("target.build.opencl").set_body_typed(BuildOpenCL);
}  // namespace codegen
}  // namespace tvm
