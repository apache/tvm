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

class InferTextureAccess : public StmtExprVisitor {
public:
  static constexpr const uint8_t read_access = 1;
  static constexpr const uint8_t write_access = 2;

  explicit InferTextureAccess() {}
  std::unordered_map<const VarNode*, std::string> Infer(const Stmt& n) {
    this->operator()(n);
    std::unordered_map<const VarNode*, std::string> storage_scope_qualifiers;
    for (auto& texture : var_access_map_) {
      if (texture.second == read_access) {
        storage_scope_qualifiers.insert({texture.first, "texture_read"});
      }
      else if (texture.second == write_access) {
        storage_scope_qualifiers.insert({texture.first, "texture_write"});
      }
      else if (texture.second == (read_access | write_access)) {
        storage_scope_qualifiers.insert({texture.first, ""});
      }
    }
    return storage_scope_qualifiers;
  }
  void VisitExpr_(const CallNode* op) {
    if (!op->args.size())
    {
      return;
    }
    if (const VarNode* buffer = op->args[0].as<VarNode>())
    {
      if (op->op.same_as(builtin::text2d_load())) {
        var_access_map_[buffer] |= read_access;
      }
      else if (op->op.same_as(builtin::text2d_store())) {
        var_access_map_[buffer] |= write_access;
      }
    }
  }
private:
  std::unordered_map<const VarNode*, uint8_t> var_access_map_;
};


CodeGenOpenCL::CodeGenOpenCL() { restrict_keyword_ = "restrict"; }

void CodeGenOpenCL::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  this->SetTextureScope(InferTextureAccess().Infer(f->body));
  for (Var arg : f->params) {
    if (arg->type_annotation.as<TextureTypeNode>())
    {
      // Storage scope qualifiers for textures are inferred
      // and set prior function codegen.
      continue;
    }
    else if (arg.dtype().is_handle()) {
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

void CodeGenOpenCL::PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    PrintType(ptr->element_type, os);
    os << '*';
  } else if (type.as<TextureTypeNode>()){
    os << "image2d_t";
  } else if (IsVoidType(type)) {
    os << "void";
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
  }
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
  } else if (scope == "texture_read") {
    os << "__read_only ";
  } else if (scope == "texture_write") {
    os << "__write_only ";
  }
}

void CodeGenOpenCL::PrintRestrict(const Var& v, std::ostream& os) {
  // Only apply restrict qualifer for non-texture types
  if (v->type_annotation.as<TextureTypeNode>() == nullptr)
  {
    os << ' ' << restrict_keyword_;
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

void CodeGenOpenCL::VisitStmt_(const StoreNode* op) {
  if (auto call = op->value.as<CallNode>()) {
    if (call->op.same_as(builtin::text2d_load())) {
      need_texture_ssa_ = false;
      // If storing a texture load into a buffer, don't use an
      // intermediate local unless the buffer allocation is a
      // single element selected from the texture read.
      auto it = allocation_size_.find(op->buffer_var.get());
      if (it != allocation_size_.end() && it->second == 1)
      {
        need_texture_ssa_ = true;
      }
    }
  }
  CodeGenC::VisitStmt_(op);
  need_texture_ssa_ = true;
}

void CodeGenOpenCL::VisitExpr_(const CastNode* op, std::ostream& os) {
  if (auto call = op->value.as<CallNode>()) {
    if (call->op.same_as(builtin::text2d_load())) {
      need_texture_ssa_ = false;
    }
  }
  CodeGenC::VisitExpr_(op, os);
  need_texture_ssa_ = true;
}

void CodeGenOpenCL::VisitStmt_(const AllocateNode* op) {
  allocation_size_.insert({op->buffer_var.get(), op->constant_allocation_size() * op->dtype.lanes()});
  CodeGenC::VisitStmt_(op);
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
    auto* texture_type  = op->args[0].as<VarNode>()->type_annotation.as<TextureTypeNode>();
    ICHECK(texture_type != nullptr) << "builtin::text2d_store() only supports storing to texture buffers";
    DataType buffer_type = texture_type->element_type.as<PrimTypeNode>()->dtype;
    if (buffer_type.is_float16()) {
      os << "write_imageh(";
    }
    else if (buffer_type.is_float()) {
      os << "write_imagef(";
    }
    this->PrintExpr(op->args[0], os);
    os << ", ";
    os << "(int2)(";
    this->PrintExpr(op->args[1], os);
    os << ", ";
    this->PrintExpr(op->args[2], os);
    os << "), ";
    this->PrintExpr(op->args[3], os);
    os << ")";
  } else if (op->op.same_as(builtin::text2d_load())) {
    std::stringstream ss;
    if (op->dtype.is_float16()) {
      ss << "read_imageh(";
    }
    else if (op->dtype.is_float()) {
      ss << "read_imagef(";
    }
    this->PrintExpr(op->args[0], ss);
    ss << ", ";
    ss << "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, ";
    ss << "(int2)(";
    this->PrintExpr(op->args[1], ss);
    ss << ", ";
    this->PrintExpr(op->args[2], ss);
    ss << "))";

    // Only use local SSA if texture is not already being stored
    if (need_texture_ssa_)
    {
      std::string rhs = SSAGetID(ss.str(), op->dtype.with_lanes(4));
      if (op->args.back().as<RampNode>())
      {
        os << rhs;
      } else {
        os << "((";
        this->PrintType(op->dtype.with_lanes(1), os);
        os << "*)&" << rhs << ")[";
        this->PrintExpr(op->args.back(), os);
        os << "]";
      }
    } else {
      os << ss.str();
    }
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

void CodeGenOpenCL::SetTextureScope(const std::unordered_map<const VarNode*, std::string>& scope) { // NOLINT(*)
  for (auto& texture : scope)
  {
    alloc_storage_scope_.insert(texture);
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
