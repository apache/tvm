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
#include "../../runtime/texture.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"
#include "../spirv/spirv_utils.h"

namespace tvm {
namespace codegen {

class InferTextureAccess : public StmtExprVisitor {
 public:
  static constexpr const uint8_t kReadAccess = 1;
  static constexpr const uint8_t kWriteAccess = 2;

  InferTextureAccess() {}
  std::unordered_map<const VarNode*, std::string> Infer(const Stmt& n) {
    StmtExprVisitor::VisitStmt(n);
    std::unordered_map<const VarNode*, std::string> storage_scope_qualifiers;
    for (auto& texture : var_access_map_) {
      if (texture.second == kReadAccess) {
        storage_scope_qualifiers.insert({texture.first, "texture_read"});
      } else if (texture.second == kWriteAccess) {
        storage_scope_qualifiers.insert({texture.first, "texture_write"});
      } else if (texture.second == (kReadAccess | kWriteAccess)) {
        storage_scope_qualifiers.insert({texture.first, ""});
      }
    }
    return storage_scope_qualifiers;
  }
  void VisitExpr_(const CallNode* op) {
    if (op->op.same_as(builtin::texture2d_load())) {
      var_access_map_[op->args[0].as<VarNode>()] |= kReadAccess;
    } else if (op->op.same_as(builtin::texture2d_store())) {
      var_access_map_[op->args[0].as<VarNode>()] |= kWriteAccess;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

 private:
  std::unordered_map<const VarNode*, uint8_t> var_access_map_;
};

CodeGenOpenCL::CodeGenOpenCL() {
  // Set OpenCL specific restrict keyword
  restrict_keyword_ = "restrict";
}

void CodeGenOpenCL::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  this->SetTextureScope(InferTextureAccess().Infer(f->body));
  for (Var arg : f->params) {
    auto ptr_type = arg->type_annotation.as<PointerTypeNode>();
    if (ptr_type && runtime::IsTextureStorage(std::string(ptr_type->storage_scope))) {
      // Storage scope qualifiers for textures are inferred
      // and set prior to function codegen.
      continue;
    } else if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

void CodeGenOpenCL::PrintFuncPrefix(std::ostream& os) { os << "__kernel "; }

void CodeGenOpenCL::PreFunctionBody(const PrimFunc& f) {
  for (Var arg : f->params) {
    auto ptr_type = arg->type_annotation.as<PointerTypeNode>();
    if (ptr_type && runtime::IsTextureStorage(std::string(ptr_type->storage_scope))) {
      this->stream << "  const sampler_t image_sampler = "
                      "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n";
      return;
    }
  }
}

std::string CodeGenOpenCL::Finish() {
  // inject extension enable pragma for fp16 and fp64
  if (enable_fp16_) {
    decl_stream << "#ifdef cl_khr_fp16\n"
                   "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                   "#elif defined(cl_amd_fp16)\n"
                   "#pragma OPENCL EXTENSION cl_amd_fp16 : enable\n"
                   "#else\n"
                   "#error \"Half precision floating point not supported"
                   " by OpenCL implementation on your device.\" \n"
                   "#endif\n\n";
  }

  if (enable_fp64_) {
    decl_stream << "#ifdef cl_khr_fp64\n"
                   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                   "#elif defined(cl_amd_fp64)\n"
                   "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
                   "#else\n"
                   "#error \"Double precision floating point not supported"
                   " by OpenCL implementation on your device.\" \n"
                   "#endif\n\n";
  }

  // Enable atomic_add used by get_valid_counts. Only needed for OpenCL < 1.1.
  if (enable_atomics_) {
    decl_stream << "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
                   "#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable\n\n";
  }

  // Enable OpenCL 1.2 sampler-less texture reads, but utilize
  // provided sampler in OpenCL 2.0.
  if (enable_compliant_texture_reads_) {
    // TODO(csullivan, lunderberg): Extend device attribute querying to support remote devices
    // generically through the device API such that a target can be created from a specific device's
    // attributes and utilized during codegen. Potential generlization of #8127 (c02cafb) for remote
    // devices.
    //
    // E.g. Only provide an image sampler when the local or remote device supports OpenCL 2.0,
    //      see below for context.
    //
    // For backwards compatibility with OpenCL 1.2, sampler-less read_image calls are used.
    // By default in sampler-less read_image calls OpenCL defaults to
    // sampler_ = "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST";
    // See section 6.12.14.3 Built-in Image Sampler-less Read Functions in the OpenCL 1.2
    // specification. For OpenCL 2.0 it can be preferable to use,
    // sampler_ = "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST";
    // For now we rely on OpenCL preprocessor directives to utilize the correct behavior
    // depending on the OpenCL version detected at OpenCL compile time.
    decl_stream << "#ifdef __OPENCL_VERSION__\n"
                << "#if __OPENCL_VERSION__ == CL_VERSION_2_0"
                << " || __OPENCL_VERSION__ == CL_VERSION_3_0 \n"
                << "#define READ_IMAGEH(image, sampler, coord) "
                << "read_imageh(image, sampler, coord)\n"
                << "#define READ_IMAGEF(image, sampler, coord) "
                << "read_imagef(image, sampler, coord)\n"
                << "#else\n"
                << "#define READ_IMAGEH(image, sampler, coord) "
                << "read_imageh(image, coord)\n"
                << "#define READ_IMAGEF(image, sampler, coord) "
                << "read_imagef(image, coord)\n"
                << "#endif\n"
                << "#endif\n\n";
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
    if (runtime::IsTextureStorage(std::string(ptr->storage_scope))) {
      os << "image2d_t";
    } else {
      PrintType(ptr->element_type, os);
      os << '*';
    }
  } else if (IsVoidType(type)) {
    os << "void";
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
  }
}

void CodeGenOpenCL::PrintVecAddr(const BufferNode* buffer, DataType t, PrimExpr base,
                                 std::ostream& os) {  // NOLINT(*)
  const VarNode* buffer_var = buffer->data.get();
  if (!HandleTypeMatch(buffer_var, t.element_of())) {
    os << '(';
    auto it = alloc_storage_scope_.find(buffer_var);
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    PrintType(t.element_of(), os);
    os << "*)";
  }
  os << GetVarID(buffer_var) << " + ";
  PrintExpr(base, os);
}
std::string CodeGenOpenCL::GetVecLoad(DataType t, const BufferNode* buffer, PrimExpr base) {
  std::ostringstream os;
  os << "vload" << t.lanes() << "(0, ";
  PrintVecAddr(buffer, t, base, os);
  os << ")";
  return os.str();
}

void CodeGenOpenCL::PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
                                  const std::string& value) {
  this->PrintIndent();
  stream << "vstore" << t.lanes() << "(" << value << ", 0, ";
  PrintVecAddr(buffer, t, base, stream);
  stream << ");\n";
}

void CodeGenOpenCL::PrintVecElemLoadExpr(DataType t, int i, const std::string& value,
                                         std::ostream& os) {  // NOLINT(*)
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (i != 0) {
      os << "|";
    }
    os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
    return;
  }
  if (i == 0) {
    // NOTE: opencl print things as (float2)(v0, v1)
    os << "((";
    PrintType(t, os);
    os << ")(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << "))";
  }
  return;
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
  // Apply restrict qualifer for non-texture types only
  if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
    if (!runtime::IsTextureStorage(std::string(ptr->storage_scope))) {
      os << ' ' << restrict_keyword_;
    }
  }
}

std::string CodeGenOpenCL::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  return CastTo(value, target);
}

std::string CodeGenOpenCL::CastTo(std::string value, DataType target) {
  std::ostringstream os;
  if (target == DataType::Bool()) {
    os << "(";
    os << "(";
    this->PrintType(target, os);
    os << ")" << value << ")";
    return os.str();
  } else {
    os << "(";
    os << "convert_";
    this->PrintType(target, os);
    os << "(" << value << "))";
    return os.str();
  }
}

void CodeGenOpenCL::VisitStmt_(const AllocateNode* op) {
  allocation_size_.insert({op->buffer_var.get(), op->ConstantAllocationSize() * op->dtype.lanes()});
  CodeGenC::VisitStmt_(op);
}

void CodeGenOpenCL::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (op->op.same_as(builtin::address_of())) {
    // Overload tvm_address_of to add storage scope (e.g. __global).
    const BufferLoadNode* load = op->args[0].as<BufferLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    ICHECK_EQ(load->indices.size(), 1) << "CodeGenOpenCL only supports flat memory allocations.";
    os << "((";
    auto it = alloc_storage_scope_.find(load->buffer->data.get());
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    this->PrintType(load->dtype.element_of(), os);
    os << " *)" << this->GetVarID(load->buffer->data.get()) << " + ";
    this->PrintExpr(load->indices[0], os);
    os << ')';
  } else if (op->op.same_as(builtin::texture2d_store())) {
    auto* ptr_type = op->args[0].as<VarNode>()->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr_type != nullptr) << "Texture Var's must be of PointerType";
    ICHECK(runtime::IsTextureStorage(std::string(ptr_type->storage_scope)))
        << "builtin::texture2d_store() only supports storing to texture buffers";
    DataType buffer_type = ptr_type->element_type.as<PrimTypeNode>()->dtype;
    if (buffer_type.is_float16()) {
      os << "write_imageh(";
    } else if (buffer_type.is_float()) {
      os << "write_imagef(";
    } else {
      LOG(FATAL) << "Unsupported type: " << buffer_type
                 << ", currently only float and half are supported for image2d OpenCL codegen.";
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
  } else if (op->op.same_as(builtin::texture2d_load())) {
    enable_compliant_texture_reads_ = true;
    std::stringstream ss;
    if (op->dtype.is_float16()) {
      ss << "READ_IMAGEH(";
    } else if (op->dtype.is_float()) {
      ss << "READ_IMAGEF(";
    } else {
      LOG(FATAL) << "Unsupported type: " << op->dtype
                 << ", currently only float and half are supported for image2d OpenCL codegen.";
    }
    this->PrintExpr(op->args[0], ss);
    ss << ", ";
    ss << "image_sampler, ";
    ss << "((int2)(";
    this->PrintExpr(op->args[1], ss);
    ss << ", ";
    this->PrintExpr(op->args[2], ss);
    ss << ")))";

    std::string rhs = SSAGetID(ss.str(), op->dtype.with_lanes(4));
    if (op->args.back().as<RampNode>()) {
      os << rhs;
    } else {
      os << "((";
      this->PrintType(op->dtype.with_lanes(1), os);
      os << "*)&" << rhs << ")[";
      this->PrintExpr(op->args.back(), os);
      os << "]";
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

void CodeGenOpenCL::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
  os << "((";
  PrintType(op->dtype, os);
  os << ")(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != op->lanes - 1) os << ", ";
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

template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr, std::ostream& os, CodeGenOpenCL* p) {
  if (op->dtype.lanes() == 1) {
    os << opstr << "((";
    p->PrintType(op->a->dtype, os);
    os << ")";
    p->PrintExpr(op->a, os);
    os << ", (";
    p->PrintType(op->b->dtype, os);
    os << ")";
    p->PrintExpr(op->b, os);
    os << ')';
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

void CodeGenOpenCL::VisitExpr_(const MinNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "min", os, this);
}

void CodeGenOpenCL::VisitExpr_(const MaxNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "max", os, this);
}

void CodeGenOpenCL::VisitExpr_(const AndNode* op, std::ostream& os) {
  std::ostringstream oss;
  os << "(";
  this->PrintExpr(op->a, oss);
  os << CastTo(oss.str(), op->dtype);
  oss.str("");
  os << " && ";
  this->PrintExpr(op->b, oss);
  os << CastTo(oss.str(), op->dtype);
  os << ")";
}

void CodeGenOpenCL::VisitExpr_(const OrNode* op, std::ostream& os) {
  std::ostringstream oss;
  os << "(";
  this->PrintExpr(op->a, oss);
  os << CastTo(oss.str(), op->dtype);
  oss.str("");
  os << " || ";
  this->PrintExpr(op->b, oss);
  os << CastTo(oss.str(), op->dtype);
  os << ")";
}

void CodeGenOpenCL::VisitExpr_(const SelectNode* op, std::ostream& os) {
  std::ostringstream oss;
  os << "select(";
  PrintExpr(op->false_value, oss);
  os << CastFromTo(oss.str(), op->false_value.dtype(), op->dtype);
  oss.str("");
  os << ", ";
  PrintExpr(op->true_value, oss);
  os << CastFromTo(oss.str(), op->true_value.dtype(), op->dtype);
  oss.str("");
  os << ", ";
  PrintExpr(op->condition, oss);
  if (op->dtype.is_float()) {
    os << CastTo(oss.str(), DataType::Int(op->dtype.bits(), op->dtype.lanes()));
  } else {
    os << CastFromTo(oss.str(), op->condition.dtype(), op->dtype);
  }
  os << ")";
}

void CodeGenOpenCL::SetTextureScope(
    const std::unordered_map<const VarNode*, std::string>& scope) {  // NOLINT(*)
  for (auto& texture : scope) {
    alloc_storage_scope_.insert(texture);
  }
}

runtime::Module BuildOpenCL(IRModule mod, Target target) {
#if TVM_ENABLE_SPIRV
  Optional<String> device = target->GetAttr<String>("device");
  if (device && device.value() == "spirv") {
    auto [smap, spirv_text] = LowerToSPIRV(mod, target);
    return runtime::OpenCLModuleCreate(smap, spirv_text, ExtractFuncInfo(mod));
  }
#endif

  using tvm::runtime::Registry;
  bool output_ssa = false;

  Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenOpenCL: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    auto calling_conv = prim_func->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenOpenCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    functions.Set(gvar, prim_func);
  }

  std::stringstream code;
  const auto* fpostproc = Registry::Get("tvm_callback_opencl_postproc");
  for (auto [gvar, prim_func] : functions) {
    code << "// Function: " << gvar->name_hint << std::endl;
    CodeGenOpenCL cg;
    cg.Init(output_ssa);
    for (auto [other_gvar, other_prim_func] : functions) {
      cg.DeclareFunction(other_gvar, other_prim_func);
    }
    cg.AddFunction(gvar, prim_func);
    std::string fsource = cg.Finish();
    if (fpostproc) {
      fsource = (*fpostproc)(fsource, target).operator std::string();
    }
    code << fsource;
  }

  return OpenCLModuleCreate(code.str(), "cl", ExtractFuncInfo(mod), code.str());
}

TVM_REGISTER_GLOBAL("target.build.opencl").set_body_typed(BuildOpenCL);
}  // namespace codegen
}  // namespace tvm
