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
 * \file codegen_cuda.cc
 */

#include "codegen_cuda.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/stmt_functor.h>

#include <cmath>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

#include "../../../runtime/thread_storage_scope.h"
#include "../../../target/build_common.h"
#include "../../../tirx/transform/ir_utils.h"
#include "cuda_fallback_module.h"
#include "literal/cuda_half_t.h"
#include "literal/cuda_int8_t.h"
#include "ptx.h"

namespace tvm {
namespace codegen {

namespace {

bool IsOp(const CallNode* call, const Op& compat_op, const char* canonical_name) {
  if (call->op.same_as(compat_op)) {
    return true;
  }
  const auto* op_node = call->op.as<OpNode>();
  return op_node != nullptr && op_node->name == canonical_name;
}

TVM_FFI_INLINE bool IsFloat8(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat8_e3m4, DLDataTypeCode::kDLFloat8_e4m3,
                        DLDataTypeCode::kDLFloat8_e4m3b11fnuz, DLDataTypeCode::kDLFloat8_e4m3fn,
                        DLDataTypeCode::kDLFloat8_e4m3fnuz, DLDataTypeCode::kDLFloat8_e5m2,
                        DLDataTypeCode::kDLFloat8_e5m2fnuz, DLDataTypeCode::kDLFloat8_e8m0fnu);
}

TVM_FFI_INLINE bool IsFloat6(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat6_e2m3fn, DLDataTypeCode::kDLFloat6_e3m2fn);
}

TVM_FFI_INLINE bool IsFloat4(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn);
}

TVM_FFI_INLINE bool IsPackedFloat(const PrimType& ty) {
  return IsFloat8(ty) || IsFloat6(ty) || IsFloat4(ty);
}

}  // namespace

std::string GetFP8Type(const PrimType& type_ty) {
  std::stringstream stream;
  int32_t lanes = type_ty.lanes();
  std::string vec;
  if (type_ty.IsScalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    TVM_FFI_THROW(InternalError)
        << "Only support scalar and vector types of width (2, 4, 8, 16) for FP8";
  }
  stream << "__nv_fp8";
  std::string suffix;
  if (type_ty.MatchesCode(DLDataTypeCode::kDLFloat8_e4m3fn)) {
    suffix = "_e4m3";
  } else if (type_ty.MatchesCode(DLDataTypeCode::kDLFloat8_e5m2)) {
    suffix = "_e5m2";
  } else if (type_ty.MatchesCode(DLDataTypeCode::kDLFloat8_e8m0fnu)) {
    suffix = "_e8m0";
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported FP8 type in CUDA codegen";
  }
  stream << vec << suffix;
  return stream.str();
}

std::string GetFP6Type(const PrimType& type_ty) {
  std::stringstream stream;
  int32_t lanes = type_ty.lanes();
  std::string vec;
  if (type_ty.IsScalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    TVM_FFI_THROW(InternalError) << "Only support scalar and vector types of width (2, 4) for FP6";
  }
  stream << "__nv_fp6";
  std::string suffix;
  if (type_ty.MatchesCode(DLDataTypeCode::kDLFloat6_e2m3fn)) {
    suffix = "_e2m3";
  } else if (type_ty.MatchesCode(DLDataTypeCode::kDLFloat6_e3m2fn)) {
    suffix = "_e3m2";
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported FP6 type in CUDA codegen";
  }
  stream << vec << suffix;
  return stream.str();
}

std::string GetFP4Type(const PrimType& type_ty) {
  std::stringstream stream;
  int32_t lanes = type_ty.lanes();
  std::string vec;
  if (type_ty.IsScalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    TVM_FFI_THROW(InternalError) << "Only support scalar and vector types of width (2, 4) for FP4";
  }
  stream << "__nv_fp4";
  std::string suffix;
  if (type_ty.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn)) {
    suffix = "_e2m1";
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported FP4 type in CUDA codegen";
  }
  stream << vec << suffix;
  return stream.str();
}

CodeGenCUDA::CodeGenCUDA(Target target) : target(target) { restrict_keyword_ = "__restrict__"; }

void CodeGenCUDA::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  vid_global_barrier_state_ = name_supply_->FreshName(runtime::symbol::tvm_global_barrier_state);
  vid_global_barrier_expect_ = name_supply_->FreshName("__barrier_expect");
  TVM_FFI_ICHECK_EQ(vid_global_barrier_state_, runtime::symbol::tvm_global_barrier_state);
}

void CodeGenCUDA::PrintFunctionSignature(const ffi::String& function_name, const PrimFunc& func,
                                         std::ostream& os) {
  CallingConv calling_conv =
      func->GetAttr<CallingConv>(tvm::attr::kCallingConv, CallingConv::kDefault).value();
  if (calling_conv == CallingConv::kDeviceKernelLaunch) {
    os << "extern \"C\" __global__ ";
  } else if (calling_conv == CallingConv::kDefault) {
    os << "extern \"C\" __device__ ";
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported calling convention for cuda codegen: "
                                 << static_cast<int>(calling_conv);
  }
  CodeGenC::PrintFunctionSignature(function_name, func, os);
}

class ThreadIdxExtractor : public tirx::StmtVisitor {
 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = op->node.as_or_throw<IterVar>();
      if (iv->var->name_hint == "threadIdx.x" || iv->thread_tag == "threadIdx.x") {
        threadIdx_x_ext = op->value;
      }
      if (iv->var->name_hint == "threadIdx.y" || iv->thread_tag == "threadIdx.y") {
        threadIdx_y_ext = op->value;
      }
      if (iv->var->name_hint == "threadIdx.z" || iv->thread_tag == "threadIdx.z") {
        threadIdx_z_ext = op->value;
      }
      if (iv->var->name_hint == "clusterCtaIdx.x" || iv->thread_tag == "clusterCtaIdx.x") {
        clusterCtaIdx_x_ext = op->value;
      }
      if (iv->var->name_hint == "clusterCtaIdx.y" || iv->thread_tag == "clusterCtaIdx.y") {
        clusterCtaIdx_y_ext = op->value;
      }
      if (iv->var->name_hint == "clusterCtaIdx.z" || iv->thread_tag == "clusterCtaIdx.z") {
        clusterCtaIdx_z_ext = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

 public:
  PrimExpr threadIdx_x_ext = IntImm::Int32(1);
  PrimExpr threadIdx_y_ext = IntImm::Int32(1);
  PrimExpr threadIdx_z_ext = IntImm::Int32(1);
  PrimExpr clusterCtaIdx_x_ext = IntImm::Int32(1);
  PrimExpr clusterCtaIdx_y_ext = IntImm::Int32(1);
  PrimExpr clusterCtaIdx_z_ext = IntImm::Int32(1);
};

void CodeGenCUDA::PrintExtraAttrs(const PrimFunc& f, std::ostream& os) {
  ThreadIdxExtractor extractor;
  extractor(f->body);
  arith::Analyzer analyzer;
  PrimExpr threadIdx_ext = analyzer->Simplify(
      extractor.threadIdx_x_ext * extractor.threadIdx_y_ext * extractor.threadIdx_z_ext);
  PrimExpr cluster_cta_yz_ext =
      analyzer->Simplify(extractor.clusterCtaIdx_y_ext * extractor.clusterCtaIdx_z_ext);
  if (const IntImmNode* const cluster_cta_yz_ext_int = cluster_cta_yz_ext.as<IntImmNode>()) {
    cluster_cta_x_is_linear_rank_ = cluster_cta_yz_ext_int->value == 1;
  } else {
    cluster_cta_x_is_linear_rank_ = false;
  }
  if (const IntImmNode* const threadIdx_ext_int = threadIdx_ext.as<IntImmNode>()) {
    if (threadIdx_ext_int->value == 1) {
      // unable to extract the number of threads per block, hence directly return
      return;
    }
    auto min_blocks_per_sm = f->GetAttr<int64_t>(tirx::attr::kLaunchBoundsMinBlocksPerSM);
    if (min_blocks_per_sm.has_value()) {
      TVM_FFI_ICHECK_GT(min_blocks_per_sm.value(), 0);
      os << " __launch_bounds__(" << threadIdx_ext_int->value << ", " << min_blocks_per_sm.value()
         << ")";
    } else {
      os << " __launch_bounds__(" << threadIdx_ext_int->value << ")";
    }
  }
}

std::string CodeGenCUDA::Finish() {
  // Generate header
  auto header_generator = ffi::Function::GetGlobal("tirx.intrinsics.cuda.header_generator");
  TVM_FFI_ICHECK(header_generator.has_value())
      << "tirx.intrinsics.cuda.header_generator is not defined";
  ffi::Array<ffi::String> tags;
  for (const auto& tag : codegen_tags_) tags.push_back(ffi::String(tag));
  std::string header = header_generator.value()(tags).cast<ffi::String>().operator std::string();
  decl_stream << header;

  // Generate util functions
  for (const auto& [name, code] : util_funcs_) {
    decl_stream << code;
  }

  return CodeGenC::Finish();
}

void CodeGenCUDA::VisitStmt_(const tirx::ForNode* op) {
  if (op->annotations.count("disable_unroll")) {
    PrintIndent();
    stream << "#pragma unroll 1\n";
  } else if (op->kind == tirx::ForKind::kUnrolled || op->annotations.count("pragma_unroll")) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::VisitStmt_(const WhileNode* op) {
  PrintIndent();
  stream << "while (1) {\n";
  int while_scope = BeginScope();
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  stream << "if (!(" << cond << ")) { break; }\n";
  PrintStmt(op->body);
  this->EndScope(while_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenCUDA::BindThreadIndex(const IterVar& iv) {
  TVM_FFI_ICHECK(!var_idmap_.count(iv->var.get()));
  const auto& scope = runtime::ThreadScope::Create(iv->thread_tag);
  if (scope.IsClusterCtaIdx()) {
    TVM_FFI_ICHECK_GE(scope.dim_index, 0);
    TVM_FFI_ICHECK_LT(scope.dim_index, 3);
    const char dim = static_cast<char>('x' + scope.dim_index);
    const std::string sreg = (scope.dim_index == 0 && cluster_cta_x_is_linear_rank_)
                                 ? "cluster_ctarank"
                                 : "cluster_ctaid." + std::string(1, dim);
    const std::string func_name = std::string("tvm_builtin_cluster_ctaid_") + dim;
    AddUtilFunction(func_name, "__forceinline__ __device__ unsigned int " + func_name +
                                   "() {\n"
                                   "  unsigned int ctaid;\n"
                                   "  asm volatile(\"mov.u32 %0, %%" +
                                   sreg +
                                   ";\" : \"=r\"(ctaid) :);\n"
                                   "  return ctaid;\n"
                                   "}\n");
    var_idmap_[iv->var.get()] = CastFromTo(func_name + "()", PrimType::UInt(32), iv->var.ty());
  } else {
    var_idmap_[iv->var.get()] = CastFromTo(iv->thread_tag, PrimType::UInt(32), iv->var.ty());
  }
}

void CodeGenCUDA::PrintType(const PrimType& t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.IsVoid()) {
    os << "void";
    return;
  }

  bool fail = false;
  if (t.MatchesCode(DLDataTypeCode::kDLFloat)) {
    switch (t.bits()) {
      case 16:
        codegen_tags_.insert("fp16");
        if (t.IsScalar()) {
          os << "half";
        } else if (lanes <= 8) {
          TVM_FFI_ICHECK_EQ(lanes % 2, 0) << "Only support an even number of lanes for half type";
          if (lanes <= 4) {
            os << "half" << lanes;
          } else {
            os << "uint" << lanes / 2;
          }
        } else {
          fail = true;
        }
        break;
      case 32:
        if (lanes <= 4) {
          os << "float";
        } else if (lanes <= 8) {
          // Emit CUDA code to access fp32 vector elements for 4 < lanes <= 8.
          //
          // float8 is stored as ulonglong4
          //
          // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
          // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
          //
          TVM_FFI_ICHECK_EQ(lanes % 2, 0) << "only support even lane for float type with lanes > 4";
          os << "ulonglong" << lanes / 2;
        } else {
          fail = true;
        }
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && (t.IsScalar() || t.bits() == 16)) return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32)) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
    codegen_tags_.insert("bf16");
    if (t.IsScalar()) {
      os << "nv_bfloat16";
    } else if (lanes <= 8) {
      TVM_FFI_ICHECK_EQ(lanes % 2, 0) << "only support even lane for bfloat16 type";
      if (lanes <= 4) {
        os << "nv_bfloat16" << lanes;
      } else {
        os << "uint" << lanes / 2;
      }
    } else {
      fail = true;
    }
    if (!fail) return;
  } else if (IsFloat8(t)) {
    codegen_tags_.insert("fp8");
    if (lanes <= 4) {
      os << GetFP8Type(t);
    } else {
      os << "uint" << lanes / 4;
    }
    return;
  } else if (IsFloat6(t)) {
    codegen_tags_.insert("fp6");
    if (lanes <= 4) {
      os << GetFP6Type(t);
    } else {
      fail = true;
    }
    return;
  } else if (IsFloat4(t)) {
    codegen_tags_.insert("fp4");
    if (lanes <= 4) {
      os << GetFP4Type(t);
    } else {
      fail = true;
    }
    return;
  } else if (t == PrimType::Bool()) {
    os << "bool";
    return;
  } else if (t.MatchesCode(DLDataTypeCode::kDLBool) && lanes > 1) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = lanes;
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.MatchesCode(DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLInt)) {
    if (t.MatchesCode(DLDataTypeCode::kDLUInt)) {
      os << "u";
    }
    switch (t.bits()) {
      case 1: {
        if (t.IsScalar()) {
          os << "int";
          return;
        } else if (lanes == 8) {
          os << "int8_t";
          return;
        } else if (lanes == 16) {
          os << "int16_t";
          return;
        } else if (lanes == 32) {
          os << "int";
          return;
        } else {
          TVM_FFI_THROW(InternalError) << "Cannot convert type " << t << " to CUDA type!";
        }
      }
      case 4: {
        if (t.IsScalar()) {
          os << "int";
          return;
        } else if (lanes == 4) {
          os << "int16_t";
          return;
        } else if (lanes == 8) {
          // directly 8 4-bit int in integer.
          os << "int";
          return;
        } else if (lanes == 16) {
          os << "int2";
          return;
        } else if (lanes == 32) {
          os << "int4";
          return;
        } else if (lanes == 64) {
          os << "int8";
          return;
        } else {
          TVM_FFI_THROW(InternalError) << "Cannot convert type " << t << " to CUDA type!";
        }
      }
      case 8: {
        if (lanes == 4) {
          // directly 4 8 bit int in integer.
          codegen_tags_.insert("int8");

          // We use int for int8x4 instead of char4 because using char4 is
          // likely to produce extra instructions to pack four int8 elements
          // into 32-bit data.
          os << "int";
          return;
        } else if (lanes == 8) {
          codegen_tags_.insert("int8");
          os << "int2";
          return;
        } else if (lanes == 16) {
          codegen_tags_.insert("int8");
          os << "int4";
          return;
        } else if (!t.MatchesCode(DLDataTypeCode::kDLUInt) && t.IsScalar()) {
          os << "signed char";
          break;
        } else {
          os << "char";
          break;
        }
      }
      case 16: {
        if (t.IsScalar()) {
          os << "short";
        } else if (lanes <= 4) {
          os << "short" << lanes;
        } else if (lanes <= 8) {
          // Emit CUDA code to access int16 vector elements.
          //
          // short4 is stored as int2
          //
          // s4.x is emitted as *(short2*)(&(i2.x)).x
          // s4.y is emitted as *(short2*)(&(i2.x)).y
          // s4.z is emitted as *(short2*)(&(i2.y)).x
          // s4.w is emitted as *(short2*)(&(i2.y)).y
          //
          TVM_FFI_ICHECK_EQ(lanes % 2, 0) << "only support even lane for shorT type with lanes > 4";
          os << "int" << lanes / 2;
        } else {
          fail = true;
        }
        if (!fail) {
          return;
        }
        break;
      }
      case 32: {
        if (t.IsScalar()) {
          os << "int";
        } else if (lanes <= 4) {
          os << "int" << lanes;
        } else if (lanes <= 8) {
          // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
          //
          // int8 is stored as longlong4
          //
          // i8.v1 is emitted as *(int2*)(&(l4.x)).x
          // i8.v2 is emitted as *(int2*)(&(l4.x)).y
          //
          TVM_FFI_ICHECK_EQ(lanes % 2, 0) << "only support even lane for int32 type with lanes > 4";
          os << "longlong" << lanes / 2;
        } else {
          fail = true;
        }
        if (!fail) {
          return;
        }
        break;
      }
      case 64: {
        if (t.IsScalar()) {
          os << "int64_t";
        } else if (lanes == 2) {
          os << "longlong2";
        } else if (lanes == 3) {
          os << "longlong3";
        } else if (lanes == 4) {
          os << "longlong4";
        }
        return;
      }
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  TVM_FFI_THROW(InternalError) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenCUDA::PrintVecConstructor(const PrimType& t, std::ostream& os) {
  os << "make_";
  PrintType(t, os);
}

void CodeGenCUDA::PrintVecBinaryOp(const std::string& op, const PrimType& t, PrimExpr lhs,
                                   PrimExpr rhs, std::ostream& os) {  // NOLINT(*)
  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    // Unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.ty());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.ty());

    for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.ty(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.ty(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.ty(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.ty(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenCUDA::PrintVecElemLoad(const std::string& vec, const PrimType& t, int i,
                                   std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.IsScalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  TVM_FFI_ICHECK(i >= 0 && i < (t.bits() == 8 ? 16 : (t.bits() == 16 || t.bits() == 32) ? 8 : 4));
  if (t.bits() == 8 && (t.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt))) {
    std::string type_name = t.MatchesCode(DLDataTypeCode::kDLInt) ? "signed char" : "unsigned char";
    if (lanes == 2 || lanes == 3) {
      os << vec << "." << access[i % lanes];
    } else {
      std::string ac = lanes == 4 ? vec : (vec + "." + access[i / 4]);
      os << "(reinterpret_cast<const " << type_name << "*>(&(" << ac << "))[" << (i % 4) << "])";
    }
  } else if (t.MatchesElementType(DLDataTypeCode::kDLFloat, 16)) {
    if (lanes <= 4) {
      os << vec << "." << access[i];
    } else {
      os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
    }
  } else if (t.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
    if (lanes <= 4) {
      os << vec << "." << access[i];
    } else {
      os << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
    }
  } else if (lanes > 4 && lanes <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.MatchesCode(DLDataTypeCode::kDLInt)) {
        type_name = "short";
      } else if (t.MatchesCode(DLDataTypeCode::kDLUInt)) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.MatchesCode(DLDataTypeCode::kDLInt)) {
        type_name = "int";
      } else if (t.MatchesCode(DLDataTypeCode::kDLUInt)) {
        type_name = "uint";
      } else if (t.MatchesCode(DLDataTypeCode::kDLFloat)) {
        type_name = "float";
      }
    }
    TVM_FFI_ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
  } else if (IsFloat4(t)) {
    os << "([](__nv_fp4_storage_t v) { __nv_fp4_e2m1 t; t.__x = v; return t; })((" << vec
       << ".__x >> " << i * 4 << ") & 0xF)";
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenCUDA::PrintVecElemStore(const std::string& vec, const PrimType& t, int i,
                                    const std::string& value) {
  int lanes = t.lanes();
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  TVM_FFI_ICHECK(i >= 0 && i < (t.bits() == 8 ? 16 : (t.bits() == 16 || t.bits() == 32) ? 8 : 4));
  if (t.bits() == 8 && (t.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt))) {
    if (lanes == 2 || lanes == 3) {
      stream << vec << '.' << access[i % lanes] << "="
             << "(" << value << ");\n";
    } else {
      std::string ac = lanes == 4 ? vec : (vec + "." + access[i / 4]);
      std::string type_name =
          t.MatchesCode(DLDataTypeCode::kDLInt) ? "signed char" : "unsigned char";
      stream << "reinterpret_cast<" << type_name << "*>(&(" << ac << "))[" << (i % 4) << "] = ("
             << type_name << ")(" << value << ");\n";
    }
  } else if (t.MatchesElementType(DLDataTypeCode::kDLFloat, 16)) {
    if (lanes <= 4) {
      stream << vec << "." << access[i] << " = " << value << ";\n";
    } else {
      stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2] << " = "
             << value << ";\n";
    }

  } else if (t.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
    if (lanes <= 4) {
      stream << vec << "." << access[i] << " = " << value << ";\n";
    } else {
      stream << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2]
             << " = " << value << ";\n";
    }
  } else if (lanes > 4 && lanes <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.MatchesCode(DLDataTypeCode::kDLInt)) {
        type_name = "short";
      } else if (t.MatchesCode(DLDataTypeCode::kDLUInt)) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.MatchesCode(DLDataTypeCode::kDLInt)) {
        type_name = "int";
      } else if (t.MatchesCode(DLDataTypeCode::kDLUInt)) {
        type_name = "uint";
      } else if (t.MatchesCode(DLDataTypeCode::kDLFloat)) {
        type_name = "float";
      }
    }
    TVM_FFI_ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2] << ")))->"
           << access[i % 2] << " = " << value << ";\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenCUDA::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  } else if (sync == "global") {
    if (!need_global_barrier_) {
      need_global_barrier_ = true;
      this->decl_stream << "extern \"C\" __device__ unsigned " << vid_global_barrier_state_
                        << ";\n";
    }
    // global synchronizer
    std::string is_load = PrintExpr(op->args[1]);
    std::string num_blocks = PrintExpr(op->args[2]);
    this->PrintIndent();
    // In theory only threadfence is needed
    // but we observed problems with only threadfence
    this->stream << "__threadfence_system();\n";
    this->PrintIndent();
    this->stream << "if (" << is_load << ") {\n";
    int wb = this->BeginScope();
    this->PrintIndent();
    this->stream << "atomicAdd(&" << vid_global_barrier_state_ << ", 1);\n";
    this->PrintIndent();
    std::string ptr = name_supply_->FreshName("pf");
    this->stream << "volatile unsigned* " << ptr << " = &" << vid_global_barrier_state_ << ";\n";
    this->PrintIndent();
    this->stream << vid_global_barrier_expect_ << " += " << num_blocks << ";\n";
    this->PrintIndent();
    this->stream << "while (" << ptr << "[0] < " << vid_global_barrier_expect_ << ");\n";
    this->EndScope(wb);
    this->PrintIndent();
    this->stream << "}\n";
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  }
}

void CodeGenCUDA::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  TVM_FFI_ICHECK_NE(scope, "global")
      << "Cannot allocate global memory when targeting CUDA. You must pass "
         "all global arrays as input instead";
  if (scope == "shared") {
    os << "__shared__ ";
  } else if (scope == "shared.dyn") {
    os << "extern __shared__ ";
  }
}

std::string CodeGenCUDA::CastFromTo(std::string value, const PrimType& from,
                                    const PrimType& target) {
  if (from == target) return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")";
  if (from.MatchesElementType(DLDataTypeCode::kDLFloat, 16) &&
      (target.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) && target.bits() == 8) {
    os << "(";
    if (target.MatchesCode(DLDataTypeCode::kDLUInt)) {
      os << "u";
    }
    os << "int)";
  }
  os << value << ")";
  return os.str();
}

void CodeGenCUDA::AddUtilFunction(const std::string& func_name, const std::string& code) {
  auto it = this->util_funcs_.find(func_name);
  if (it != this->util_funcs_.end()) {
    TVM_FFI_ICHECK_EQ(it->second, code)
        << "Function " << func_name << " already exists with different code";
    return;
  }
  this->util_funcs_.insert({func_name, code});
}

void CodeGenCUDA::VisitExpr_(const CastNode* op, std::ostream& os) {
  PrimType from_ty = op->value.ty();
  PrimType target_ty = op->ty.as_or_throw<PrimType>();
  TVM_FFI_ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  // Emit simple C-style type conversion.
  if (from_ty.IsScalar()) return CodeGenC::VisitExpr_(op, os);

  if (IsPackedFloat(target_ty) || IsPackedFloat(from_ty)) {
    std::ostringstream val;
    if (target_ty.MatchesCode(DLDataTypeCode::kDLBfloat) && target_ty.lanes() == 2) {
      val << "cast_to_nv_bfloat162(" << PrintExpr(op->value) << ")";
    } else {
      val << "(";
      PrintType(target_ty, val);
      val << ")(" << PrintExpr(op->value) << ")";
    }
    os << val.str();
    return;
  }

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  {
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
      std::ostringstream val;
      val << "(";
      PrintType(target_ty.WithLanes(1), val);
      val << ")(";
      PrintVecElemLoad(src, from_ty, i, val);
      val << ")";
      PrintVecElemStore(sret, target_ty, i, val.str());
    }
  }
  os << sret;
}

void CodeGenCUDA::PrintCallExtern(Type ret_type, ffi::String global_symbol,
                                  const ffi::Array<Expr>& args, bool skip_first_arg,
                                  std::ostream& os) {  // NOLINT(*)
  auto ret_prim_type = ret_type.as<PrimType>();
  if (ret_prim_type && ret_prim_type.value().IsFixedLengthVector()) {
    PrimType ret_ty = ret_prim_type.value();
    //
    // Emit an unsupported vector call
    //
    // v = intrin_f((float4*)A[0], (float4*)B[0])
    //
    // as
    //
    // float4 __ret;
    // {
    //   float4 __arg0 = ((float4*)A)[0];
    //   float4 __arg1 = ((float4*)B)[0];
    //   __ret.x = intrin_f(__arg0.x, __arg1.x);
    //   __ret.y = intrin_f(__arg0.y, __arg1.y);
    //   __ret.z = intrin_f(__arg0.z, __arg1.z);
    //   __ret.w = intrin_f(__arg0.w, __arg1.w);
    // }
    // v = __ret;
    //
    // Declare the result vector.
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(ret_ty, stream);
    stream << ' ' << sret << ";\n";
    {
      // Load arguments.
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        Expr arg = args[i];
        std::string val = SSAGetID(PrintExpr(arg), arg->ty);
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
      for (int i = 0; i < ret_ty.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0) scall << ", ";
          Type arg_type = args[arg_begin + j]->ty;
          if (auto prim_type = arg_type.as<PrimType>()) {
            PrintVecElemLoad(sargs[j], prim_type.value(), i, scall);
          } else {
            TVM_FFI_ICHECK(arg_type.as<PointerTypeNode>());
            scall << sargs[j];
          }
        }
        scall << ")";
        PrintVecElemStore(sret, ret_ty, i, scall.str());
      }
    }
    os << sret;
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg, os);
  }
}

void CodeGenCUDA::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (auto opt_call_opt = op->op.as<Op>()) {
    Op call_op = opt_call_opt.value();
    // This is only for backward compatibility with __shfl_{up/down}.
    // A macro will be used to replace *_sync calls to legacy ones.
    if (op_need_warp_shuffle_.get(call_op, false)) {
      codegen_tags_.insert("warp_shuffle");
    }
  }

  auto print_cuda_func_call = [&](const CallNode* op, std::ostream& os) {
    TVM_FFI_ICHECK_GE(op->args.size(), 2U);
    size_t num_args = op->args.size() - 2;
    std::vector<std::string> args;
    for (size_t i = 1; i < num_args + 1; i++) {
      args.push_back(this->PrintExpr(op->args[i]));
    }
    std::string source_code = op->args[num_args + 1].as<StringImmNode>()->value;
    std::string func_name = op->args[0].as<StringImmNode>()->value;
    os << func_name << "(";
    for (size_t i = 0; i < num_args; i++) {
      const auto& arg = args[i];
      os << arg;
      if (i < num_args - 1) {
        os << ", ";
      }
    }
    os << ")";
    AddUtilFunction(func_name, source_code);
  };

  if (auto opt_call_opt = op->op.as<Op>()) {
    Op call_op = opt_call_opt.value();
    auto codegen_getter = tvm::ffi::Function::GetGlobal("tirx.intrinsics.cuda.get_codegen");
    TVM_FFI_ICHECK(codegen_getter.has_value())
        << "tirx.intrinsics.cuda.get_codegen is not registered";
    // either codegen is registered or not
    auto codegen = codegen_getter.value()(call_op->name).cast<ffi::Optional<tvm::ffi::Function>>();
    if (codegen.has_value()) {
      // codegen is registered, it should return a Call to cuda_func_call
      auto func_call = codegen.value()(op->args);
      auto res = func_call.cast<ffi::Tuple<Call, ffi::Array<ffi::String>>>();
      print_cuda_func_call(res.get<0>().get(), os);
      for (const auto& tag : res.get<1>()) {
        codegen_tags_.insert(tag.operator std::string());
      }
      return;
    }
  }

  static const Op& tvm_fill_fragment_op = Op::Get("tirx.tvm_fill_fragment");
  static const Op& tvm_load_matrix_sync_op = Op::Get("tirx.tvm_load_matrix_sync");
  static const Op& tvm_store_matrix_sync_op = Op::Get("tirx.tvm_store_matrix_sync");
  static const Op& tvm_mma_sync_op = Op::Get("tirx.tvm_mma_sync");
  static const Op& tvm_bmma_sync_op = Op::Get("tirx.tvm_bmma_sync");
  static const Op& ptx_mma_op = Op::Get("tirx.ptx.mma");
  static const Op& ptx_mma_sp_op = Op::Get("tirx.ptx.mma_sp");
  static const Op& mma_store_op = Op::Get("tirx.mma_store");
  static const Op& mma_fill_op = Op::Get("tirx.mma_fill");
  static const Op& ptx_mma_legacy_op = Op::Get("tirx.ptx.mma_legacy");
  static const Op& ptx_ldmatrix_legacy_op = Op::Get("tirx.ptx.ldmatrix_legacy");
  static const Op& mma_store_legacy_op = Op::Get("tirx.mma_store_legacy");
  static const Op& mma_fill_legacy_op = Op::Get("tirx.mma_fill_legacy");
  static const Op& ptx_cp_async_bulk_op = Op::Get("tirx.ptx.cp_async_bulk");
  static const Op& ptx_cp_async_mbarrier_arrive_op = Op::Get("tirx.ptx.cp_async_mbarrier_arrive");
  static const Op& ptx_ldg32_op = Op::Get("tirx.ptx.ldg32");
  static const Op& cuda_func_call_op = Op::Get("tirx.cuda.func_call");

  if (op->op.same_as(tvm_fill_fragment_op)) {
    codegen_tags_.insert("mma");
    TVM_FFI_ICHECK_EQ(op->args.size(), 6U);
    os << "nvcuda::wmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ")";
  } else if (op->op.same_as(tvm_load_matrix_sync_op)) {
    codegen_tags_.insert("mma");
    TVM_FFI_ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::load_matrix_sync(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    os << ")";
  } else if (op->op.same_as(tvm_store_matrix_sync_op)) {
    codegen_tags_.insert("mma");
    TVM_FFI_ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::store_matrix_sync(";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[6], os);
    if (const StringImmNode* str = op->args[7].as<StringImmNode>()) {
      os << ", nvcuda::wmma::mem_" << str->value;
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid parameters";
    }
    os << ")";
  } else if (op->op.same_as(tvm_mma_sync_op)) {
    codegen_tags_.insert("mma");
    TVM_FFI_ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(tvm_bmma_sync_op)) {
    codegen_tags_.insert("mma");
    TVM_FFI_ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::bmma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (IsOp(op, ptx_mma_op, "tirx.ptx.mma")) {
    // arg 0: shape: mXnXkX
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: fp16, fp64, ...
    // arg 4: B precision: fp16, fp64, ...
    // arg 5: C precision: fp32, fp64, ...
    // arg 6: A multiplicand
    // arg 7: A multiplicand index
    // arg 8: B multiplicand
    // arg 9: B multiplicand index
    // arg 10: C accumulator
    // arg 11: C accumulator index
    // arg 12: saturate
    // arg 13: (optional) 1-bit operator (xor or and)
    TVM_FFI_ICHECK(op->args.size() == 13U || op->args.size() == 14U);
    std::string shape = op->args[0].as_or_throw<StringImm>()->value;
    std::string A_layout = op->args[1].as_or_throw<StringImm>()->value;
    std::string B_layout = op->args[2].as_or_throw<StringImm>()->value;
    std::string A_dtype = op->args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = op->args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = op->args[5].as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_bias = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_bias = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_bias = this->PrintExpr(op->args[11]);
    bool saturate = op->args[12].as_or_throw<IntImm>()->value;
    std::string bit_op = op->args.size() > 13 ? op->args[13].as_or_throw<StringImm>()->value : "";
    std::string asm_code =
        PrintMMAAssembly(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias, b_ref,
                         b_bias, c_ref, c_bias, "", "", "", bit_op, false, saturate);

    this->stream << asm_code;
  } else if (IsOp(op, ptx_mma_sp_op, "tirx.ptx.mma_sp")) {
    // arg 0: shape: mXnXkX
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: fp16, fp32, ...
    // arg 4: B precision: fp16, fp32, ...
    // arg 5: C precision: fp16, fp32, ...
    // arg 6: A multiplicand pointer
    // arg 7: A multiplicand index
    // arg 8: B multiplicand pointer
    // arg 9: B multiplicand index
    // arg 10: C accumulator pointer
    // arg 11: C accumulator index
    // arg 12: metadata
    // arg 13: metadata index
    // arg 14: sparse_selector
    // arg 15: saturate
    TVM_FFI_ICHECK_EQ(op->args.size(), 16U);
    std::string shape = op->args[0].as_or_throw<StringImm>()->value;
    std::string A_layout = op->args[1].as_or_throw<StringImm>()->value;
    std::string B_layout = op->args[2].as_or_throw<StringImm>()->value;
    std::string A_dtype = op->args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = op->args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = op->args[5].as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_offset = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_offset = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_offset = this->PrintExpr(op->args[11]);
    std::string metadata = this->PrintExpr(op->args[12]);
    std::string metadata_offset = this->PrintExpr(op->args[13]);
    std::string sparse_selector = this->PrintExpr(op->args[14]);
    bool saturate = op->args[15].as_or_throw<IntImm>()->value;
    std::string asm_code = PrintMMAAssembly(
        shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_offset, b_ref, b_offset,
        c_ref, c_offset, metadata, metadata_offset, sparse_selector, "", true, saturate);
    this->stream << asm_code;
  } else if (op->op.same_as(mma_store_op)) {
    int m = op->args[0].as_or_throw<IntImm>()->value;
    int n = op->args[1].as_or_throw<IntImm>()->value;
    std::string dst = this->PrintExpr(op->args[2]);
    std::string src = this->PrintExpr(op->args[3]);
    std::string src_offset = this->PrintExpr(op->args[4]);
    PrimExpr stride = op->args[5].as_or_throw<PrimExpr>();

    TVM_FFI_ICHECK(m == 16 && n == 16) << "Only m == 16 && n == 16 case supported for now";

    // Each thread in a warp holds a certain number of elements of an MMA output.
    // For example, if we compute a 16x16 tile using MMA, each thread holds 8 elements
    // in its registers. So conceptually, a warp memory is organized as a 32x8 block.
    // A map from a 16x16 tile to a 32x8 block of memory is specified by the index map below.

    // To store the 32x8 output back to a 16x16 tile in shared or global memory, we invert this map
    // to determine the output location for each 8 element.

    const auto index_map_func =
        tvm::ffi::Function::GetGlobal("tirx.index_map.shared_16x16_to_ldmatrix_32x8_layout");
    TVM_FFI_ICHECK(index_map_func.has_value());

    arith::Analyzer analyzer;
    auto inverse_index_map =
        IndexMap::FromFunc(2, *index_map_func).Inverse({Range(0, m), Range(0, n)}, analyzer);
    auto indices_16x16 = inverse_index_map->final_indices;

    // "//" and "%" in the index map are translated to FloorDiv/Mod, but the plain Div/Mod are fine.
    // FloorDiv/Mod are supposed to be lowered before they reach codegen, so manually replace them
    // to the plain ones here.
    class LowerFloorDivMod : public ExprMutator {
     public:
      Expr VisitExpr_(const FloorDivNode* op) {
        return tirx::Div(this->VisitPrimExpr(op->a), this->VisitPrimExpr(op->b));
      }
      Expr VisitExpr_(const FloorModNode* op) {
        return tirx::Mod(this->VisitPrimExpr(op->a), this->VisitPrimExpr(op->b));
      }
    };

    auto dst_ind = LowerFloorDivMod()(indices_16x16[0] * stride + indices_16x16[1]);

    var_idmap_[inverse_index_map->initial_indices[0].get()] = "threadIdx.x";
    var_idmap_[inverse_index_map->initial_indices[1].get()] = "local_id";

    os << "for (int local_id = 0; local_id < 8; ++local_id) {\n";
    os << dst << "[" + this->PrintExpr(dst_ind) + "] = " << src << "[" << src_offset
       << " + local_id];\n";
    os << "}\n";

  } else if (op->op.same_as(mma_fill_op)) {
    std::string num_elem = this->PrintExpr(op->args[0]);
    std::string dst = this->PrintExpr(op->args[1]);
    std::string dst_offset = this->PrintExpr(op->args[2]);

    os << "for (int i = 0; i < " << num_elem << "; ++i) {\n";
    os << dst << "[" << dst_offset << " + i] = 0.0;";
    os << "}\n";
  } else if (IsOp(op, ptx_mma_legacy_op, "tirx.ptx.mma_legacy")) {
    // args: shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype,
    //       a_ptr_var, a_offset, b_ptr_var, b_offset,
    //       c_ptr_var, c_offset, saturate, [bit_op]
    codegen_tags_.insert("mma");
    TVM_FFI_ICHECK(op->args.size() == 13U || op->args.size() == 14U);
    std::string shape = op->args[0].as_or_throw<StringImm>()->value;
    std::string A_layout = op->args[1].as_or_throw<StringImm>()->value;
    std::string B_layout = op->args[2].as_or_throw<StringImm>()->value;
    std::string A_dtype = op->args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = op->args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = op->args[5].as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_bias = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_bias = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_bias = this->PrintExpr(op->args[11]);
    bool saturate = op->args[12].as_or_throw<IntImm>()->value;
    std::string bit_op = op->args.size() > 13 ? op->args[13].as_or_throw<StringImm>()->value : "";
    this->stream << PrintMMAAssembly(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref,
                                     a_bias, b_ref, b_bias, c_ref, c_bias, "", "", "", bit_op,
                                     false, saturate);
  } else if (IsOp(op, ptx_ldmatrix_legacy_op, "tirx.ptx.ldmatrix_legacy")) {
    // args: trans, num, type, local_ptr_var, local_offset, smem_ptr_var, smem_offset
    codegen_tags_.insert("mma");
    TVM_FFI_ICHECK_EQ(op->args.size(), 7U);
    // `trans` and `num` may arrive as Bool/IntImm; both Downcastable
    // to PrimExpr whose IntImmNode value tells us the literal.
    bool trans = op->args[0].as_or_throw<IntImm>()->value != 0;
    int num = op->args[1].as_or_throw<IntImm>()->value;
    std::string type_str = op->args[2].as_or_throw<StringImm>()->value;
    std::string local_ptr = this->PrintExpr(op->args[3]);
    std::string local_offset = this->PrintExpr(op->args[4]);
    std::string smem_ptr = this->PrintExpr(op->args[5]);
    PrimType res_ty = op->ty.as_or_throw<PrimType>();
    if (trans && res_ty.bits() == 8) {
      // ldmatrix can't transpose 8-bit elements (it assumes 16-bit), so
      // synthesize the equivalent manual gather loop. args[6] is the
      // shared-memory stride for this fallback.
      std::string smem_stride = this->PrintExpr(op->args[6]);
      TVM_FFI_ICHECK(num == 4);
      os << "for (int i = 0; i < 16; ++i) {\n";
      os << local_ptr << "[" + local_offset + " + i] = " << smem_ptr
         << "[(i % 8) / 4 * " + smem_stride + " * 16 + (threadIdx.x % 4) * 4 * " + smem_stride +
                "+ (i % 4) * " + smem_stride + " + threadIdx.x / 4 +  (i / 8) * 8];\n";
      os << "}\n";
    } else {
      std::string smem_offset = this->PrintExpr(op->args[6]);
      this->stream << PrintLoadMatrixAssembly(trans, num, type_str, local_ptr, local_offset,
                                              smem_ptr, smem_offset);
    }
  } else if (op->op.same_as(mma_store_legacy_op)) {
    // args: m, n, dst_ptr, src_ptr_var, src_offset, dst_stride
    // (dst_ptr is typically an access_ptr Call that already encodes
    // dst.elem_offset and the global pointer cast.)
    int m = op->args[0].as_or_throw<IntImm>()->value;
    int n = op->args[1].as_or_throw<IntImm>()->value;
    std::string dst = this->PrintExpr(op->args[2]);
    std::string src = this->PrintExpr(op->args[3]);
    std::string src_offset = this->PrintExpr(op->args[4]);
    PrimExpr stride = op->args[5].as_or_throw<PrimExpr>();

    TVM_FFI_ICHECK(m == 16 && n == 16) << "Only m == 16 && n == 16 case supported for now";

    const auto index_map_func =
        tvm::ffi::Function::GetGlobal("tirx.index_map.shared_16x16_to_ldmatrix_32x8_layout");
    TVM_FFI_ICHECK(index_map_func.has_value());

    arith::Analyzer analyzer;
    auto inverse_index_map =
        IndexMap::FromFunc(2, *index_map_func).Inverse({Range(0, m), Range(0, n)}, analyzer);
    auto indices_16x16 = inverse_index_map->final_indices;

    class LowerFloorDivMod : public ExprMutator {
     public:
      Expr VisitExpr_(const FloorDivNode* op) {
        return tirx::Div(this->VisitPrimExpr(op->a), this->VisitPrimExpr(op->b));
      }
      Expr VisitExpr_(const FloorModNode* op) {
        return tirx::Mod(this->VisitPrimExpr(op->a), this->VisitPrimExpr(op->b));
      }
    };

    auto dst_ind = LowerFloorDivMod()(indices_16x16[0] * stride + indices_16x16[1]);

    var_idmap_[inverse_index_map->initial_indices[0].get()] = "threadIdx.x";
    var_idmap_[inverse_index_map->initial_indices[1].get()] = "local_id";

    os << "for (int local_id = 0; local_id < 8; ++local_id) {\n";
    os << dst << "[" << this->PrintExpr(dst_ind) << "] = " << src << "[" << src_offset
       << " + local_id];\n";
    os << "}\n";
  } else if (op->op.same_as(mma_fill_legacy_op)) {
    // args: local_size, local_ptr_var, offset
    std::string num_elem = this->PrintExpr(op->args[0]);
    std::string dst = this->PrintExpr(op->args[1]);
    std::string dst_offset = this->PrintExpr(op->args[2]);
    os << "for (int i = 0; i < " << num_elem << "; ++i) {\n";
    os << dst << "[" << dst_offset << " + i] = 0.0;";
    os << "}\n";
  } else if (IsOp(op, ptx_cp_async_bulk_op, "tirx.ptx.cp_async_bulk")) {
    codegen_tags_.insert("cast_smem_ptr_to_int");
    std::string dst = this->PrintExpr(op->args[0]);
    std::string dst_offset = this->PrintExpr(op->args[1]);
    std::string src = this->PrintExpr(op->args[2]);
    std::string src_offset = this->PrintExpr(op->args[3]);
    std::string size = this->PrintExpr(op->args[4]);
    int barrier_arr_id = op->args[5].as_or_throw<IntImm>()->value;
    int barrier_id = op->args[6].as_or_throw<IntImm>()->value;
    auto it = barrier_count_.find(barrier_arr_id);
    TVM_FFI_ICHECK(it != barrier_count_.end()) << "Barrier array does not exist";
    std::string barrier_arr = barrier_name_ + "_" + std::to_string(barrier_arr_id);
    std::string barrier = barrier_arr + "[" + std::to_string(barrier_id) + "]";
    this->stream << PrintCpAsyncBulkAsm(dst, dst_offset, src, src_offset, size, barrier);
  } else if (IsOp(op, ptx_cp_async_mbarrier_arrive_op, "tirx.ptx.cp_async_mbarrier_arrive")) {
    codegen_tags_.insert("cast_smem_ptr_to_int");
    int barrier_arr_id = op->args[0].as_or_throw<IntImm>()->value;
    int barrier_id = op->args[1].as_or_throw<IntImm>()->value;
    auto it = barrier_count_.find(barrier_arr_id);
    TVM_FFI_ICHECK(it != barrier_count_.end()) << "Barrier array does not exist";
    TVM_FFI_ICHECK(barrier_id < it->second) << "Barrier id out of bounds";
    std::string barrier_arr = barrier_name_ + "_" + std::to_string(barrier_arr_id);
    std::string barrier = barrier_arr + "[" + std::to_string(barrier_id) + "]";
    this->stream << PrintCpAsyncBarrierAsm(barrier);
  } else if (IsOp(op, ptx_ldg32_op, "tirx.ptx.ldg32")) {
    /*
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        // " @p ld.global.nc.f32 %0, [%1];}\n"t
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
        : "=f"(reg)
        : "l"(addr), "r"((int)guard)
    );
    */

    // get local
    std::string reg = this->PrintExpr(op->args[0]);
    // get guard
    std::string guard = this->PrintExpr(op->args[1]);
    const BufferLoadNode* addr_buffer = op->args[2].as<BufferLoadNode>();
    std::string global_addr = this->PrintExpr(addr_buffer->indices[0]);
    std::string global_buffer = this->PrintExpr(addr_buffer->buffer->data);
    std::string local_addr = this->PrintExpr(op->args[3]);
    this->stream << "asm volatile (\n";
    this->stream << "\"{.reg .pred p;\\n\"\n";
    this->stream << "\" setp.ne.b32 p, %2, 0;\\n\"\n";
    this->stream << "\" @!p mov.b32 %0, 0;\\n\"\n";
    this->stream << "\" @p ld.global.nc.f32 %0, [%1];}\\n\"\n";
    // stream << "\" @p ld.global.nc.L2::128B.f32 %0, [%1];}\\n\"\n" ;
    stream << ": \"=f\"(" << reg << "[" << local_addr << "]"
           << ")\n";
    stream << ": \"l\"((void*)(" << global_buffer << "+" << global_addr << ")), \"r\"((int)"
           << guard << ")\n";
    stream << ");\n";
  } else if (op->op.same_as(builtin::reinterpret())) {
    auto tgt_prim_type = op->ty.as<PrimType>();
    auto src_prim_type = op->args[0]->ty.as<PrimType>();

    if (op->args[0]->ty.as<PointerTypeNode>() && tgt_prim_type &&
        tgt_prim_type.value().IsScalar() &&
        tgt_prim_type.value().MatchesCode(DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLInt) &&
        tgt_prim_type.value().bits() == 64) {
      os << "reinterpret_cast<";
      this->PrintType(tgt_prim_type.value(), os);
      os << ">(" << PrintExpr(op->args[0]) << ")";
      return;
    }

    if (!tgt_prim_type || !src_prim_type) {
      return CodeGenC::VisitExpr_(op, os);
    }

    PrimType tgt_ty = op->ty.as_or_throw<PrimType>();
    PrimExpr value = op->args[0].as_or_throw<PrimExpr>();
    PrimType src_ty = value.ty();

    // Handle float4_e2m1fn reinterpret
    if (!IsFloat4(src_ty) && !IsFloat4(tgt_ty)) {
      return CodeGenC::VisitExpr_(op, os);
    }
    if (src_ty == tgt_ty || tgt_ty.lanes() * tgt_ty.bits() == src_ty.lanes() * src_ty.bits()) {
      return CodeGenC::VisitExpr_(op, os);
    }
    TVM_FFI_ICHECK_EQ(tgt_ty.lanes(), src_ty.lanes())
        << "E2M1 float4 reinterpret expects source and target to have the same number of lanes. "
        << "Source dtype: " << src_ty << ", Target dtype: " << tgt_ty;
    TVM_FFI_ICHECK_EQ(tgt_ty.StorageBytes(), src_ty.StorageBytes())
        << "E2M1 float4 reinterpret expects source and target to have the same number of bytes. "
        << "Source dtype: " << src_ty << ", Target dtype: " << tgt_ty;

    int lanes = tgt_ty.lanes();

    int ssa_scope = BeginScope();
    if (lanes == 1) {
      // The case of lane=1 is same as the normal reinterpret,
      // except that we allow the src and dst dtype to have different number of bits.
      std::string rhs = SSAGetID(PrintExpr(value), src_ty);
      os << "(*(";
      this->PrintType(tgt_ty, os);
      os << " *)(&(" << rhs << ")))";
    } else if (lanes == 2) {
      if (IsFloat4(tgt_ty)) {
        // We view the source as an uint16, and then extract bits of two fp4 numbers,
        // and finally reinterpret the result as fp4x2.
        value =
            Call(PrimType::UInt(16), tirx::builtin::reinterpret(), {value}).as_or_throw<PrimExpr>();
        tirx::PrimVar temp_var("temp_var", PrimType::UInt(16));
        value = tirx::Let(temp_var, value,
                          tirx::Cast(PrimType::UInt(8),
                                     (temp_var & IntImm(PrimType::UInt(16), 0xF)) |
                                         ((temp_var >> 4) & IntImm(PrimType::UInt(16), 0xF0))));
      } else {
        value = tirx::Cast(
            PrimType::UInt(16),
            Call(PrimType::UInt(8), tirx::builtin::reinterpret(), {value}).as_or_throw<PrimExpr>());
        tirx::PrimVar temp_var("temp_var", PrimType::UInt(16));
        value = tirx::Let(temp_var, value,
                          (temp_var & IntImm(PrimType::UInt(16), 0xF)) |
                              ((temp_var & IntImm(PrimType::UInt(16), 0xF0)) << 4));
      }
      os << PrintExpr(Call(tgt_ty, tirx::builtin::reinterpret(), {value}).as_or_throw<PrimExpr>());
    } else if (lanes == 4) {
      if (IsFloat4(tgt_ty)) {
        // We view the source as an uint32, and then extract bits of four fp4 numbers,
        // and finally reinterpret the result as fp4x4.
        value =
            Call(PrimType::UInt(32), tirx::builtin::reinterpret(), {value}).as_or_throw<PrimExpr>();
        tirx::PrimVar temp_var("temp_var", PrimType::UInt(32));
        value = tirx::Let(temp_var, value,
                          tirx::Cast(PrimType::UInt(16),
                                     (temp_var & IntImm(PrimType::UInt(32), 0xF)) |
                                         ((temp_var >> 4) & IntImm(PrimType::UInt(32), 0xF0)) |
                                         ((temp_var >> 8) & IntImm(PrimType::UInt(32), 0xF00)) |
                                         ((temp_var >> 12) & IntImm(PrimType::UInt(32), 0xF000))));
      } else {
        value = tirx::Cast(PrimType::UInt(32),
                           Call(PrimType::UInt(16), tirx::builtin::reinterpret(), {value})
                               .as_or_throw<PrimExpr>());
        tirx::PrimVar temp_var("temp_var", PrimType::UInt(32));
        value = tirx::Let(temp_var, value,
                          (temp_var & IntImm(PrimType::UInt(32), 0xF)) |
                              ((temp_var & IntImm(PrimType::UInt(32), 0xF0)) << 4) |
                              ((temp_var & IntImm(PrimType::UInt(32), 0xF00)) << 8) |
                              ((temp_var & IntImm(PrimType::UInt(32), 0xF000)) << 12));
      }
      os << PrintExpr(Call(tgt_ty, tirx::builtin::reinterpret(), {value}).as_or_throw<PrimExpr>());
    } else {
      TVM_FFI_THROW(InternalError)
          << "Invalid number of lanes for float4_e2m1fn reinterpret: " << lanes;
    }
    EndScope(ssa_scope);
  } else if (op->op.same_as(builtin::print_buffer())) {
    TVM_FFI_ICHECK_GE(op->args.size(), 5U) << "Print operation expects at least 5 arguments";

    Expr arg = op->args[0];
    const auto* var_node = arg.as<VarNode>();
    PrimType dtype_ty = op->ty.as_or_throw<PrimType>();
    bool is_string = op->args[2].as<IntImmNode>()->value;
    bool is_scalar = op->args[3].as<IntImmNode>()->value;
    int num_dims = op->args[4].as<IntImmNode>()->value;

    TVM_FFI_ICHECK(!(is_string && is_scalar)) << "Cannot have both is_string and is_scalar true";
    if (is_string) {
      // String printing logic
      std::string print_arg = var_node ? GetVarID(var_node) : PrintExpr(arg);
      std::string buffer_name = var_node ? GetVarID(var_node) : "string_literal";
      os << "// print_buffer starts (string)\n"
         << "if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {\n"
         << "  printf(\"" << buffer_name << ": %s\\n\\n\", (char*)" << print_arg << ");\n"
         << "}\n"
         << "// print_buffer ends\n";
      return;
    }

    if (is_scalar) {
      // Scalar printing logic
      std::string format_specifier;
      bool is_float16 = dtype_ty.MatchesElementType(DLDataTypeCode::kDLFloat, 16);
      if (dtype_ty.MatchesCode(DLDataTypeCode::kDLFloat))
        format_specifier = "%f";
      else if (dtype_ty.MatchesCode(DLDataTypeCode::kDLInt))
        format_specifier = "%d";
      else if (dtype_ty.MatchesCode(DLDataTypeCode::kDLUInt))
        format_specifier = "%u";
      else
        TVM_FFI_THROW(InternalError) << "Unsupported data type for scalar print: " << dtype_ty;

      std::string print_arg = var_node ? ("*" + GetVarID(var_node)) : PrintExpr(arg);
      os << "// print_buffer starts (scalar)\n"
         << "if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {\n"
         << "  printf(\"Scalar (dtype: " << dtype_ty << "): " << format_specifier << "\\n\\n\", "
         << (is_float16 ? "static_cast<float>(" : "") << print_arg << (is_float16 ? ")" : "")
         << ");\n"
         << "}\n"
         << "// print_buffer ends\n";
      return;
    }

    Array<PrimExpr> shape;
    for (size_t i = 5; i < op->args.size(); ++i) {
      shape.push_back(op->args[i].as_or_throw<PrimExpr>());
    }

    std::string format_specifier;
    bool is_float16 = false;
    if (dtype_ty.MatchesCode(DLDataTypeCode::kDLFloat)) {
      if (dtype_ty.bits() == 16) {
        format_specifier = "%f";
        is_float16 = true;
      } else {
        format_specifier = "%f";
      }
    } else if (dtype_ty.MatchesCode(DLDataTypeCode::kDLInt)) {
      format_specifier = "%d";
    } else if (dtype_ty.MatchesCode(DLDataTypeCode::kDLUInt)) {
      format_specifier = "%u";
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported data type for print: " << dtype_ty;
    }

    TVM_FFI_ICHECK(var_node) << "Formatted print is only supported for buffer variables.";
    std::string buffer_name = GetVarID(var_node);

    os << "// print_buffer starts (buffer)\n"
       << "if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {\n";

    os << "  printf(\"(" << buffer_name << ", shape=(";
    for (int i = 0; i < num_dims; ++i) {
      os << PrintExpr(shape[i]) << (i < num_dims - 1 ? "," : "");
    }
    os << "), dtype=" << dtype_ty << "):\\n\");\n";

    std::vector<std::string> loop_vars;
    for (int i = 0; i < num_dims; ++i) {
      loop_vars.push_back("i" + std::to_string(i));
    }

    std::function<void(int)> GenerateLoops;
    GenerateLoops = [&](int dim) {
      if (dim == num_dims) {
        std::string idx_calculation;
        if (num_dims > 0) {
          idx_calculation = loop_vars[0];
          for (int i = 1; i < num_dims; ++i) {
            idx_calculation =
                "(" + idx_calculation + " * " + PrintExpr(shape[i]) + " + " + loop_vars[i] + ")";
          }
        } else {
          idx_calculation = "0";
        }

        os << std::string(num_dims * 2 + 4, ' ') << "printf(\"" << format_specifier << "\", ";
        if (is_float16) {
          os << "static_cast<float>(" << buffer_name << "[" << idx_calculation << "]));\n";
        } else {
          os << buffer_name << "[" << idx_calculation << "]);\n";
        }
        return;
      }

      std::string indent(dim * 2 + 2, ' ');
      os << indent << "for (int " << loop_vars[dim] << " = 0; " << loop_vars[dim] << " < "
         << PrintExpr(shape[dim]) << "; ++" << loop_vars[dim] << ") {\n";

      if (dim < num_dims - 1) {
        os << indent << "  printf(\"[\");\n";
      }
      GenerateLoops(dim + 1);

      if (dim < num_dims - 1) {
        os << indent << "  printf(\"]\");\n";
      }

      os << indent << "  if (" << loop_vars[dim] << " < " << PrintExpr(shape[dim]) << " - 1) {\n";
      if (dim == num_dims - 1) {
        os << indent << "    printf(\" \");\n";
      } else {
        os << indent << "    printf(\"\\n" << std::string(dim + 2, ' ') << "\");\n";
      }
      os << indent << "  }\n";

      os << indent << "}\n";
    };

    os << "  printf(\"[\");\n";
    if (num_dims > 0) {
      GenerateLoops(0);
    }
    os << "  printf(\"]\\n\");\n";

    os << "}\n"
       << "// print_buffer ends\n";
  } else if (op->op.same_as(cuda_func_call_op) ||
             (op->op.as<Op>() && op->op.as<Op>().value()->name == "tirx.cuda.func_call")) {
    print_cuda_func_call(op, os);
  } else if (op->op.same_as(builtin::thread_return())) {
    os << "return";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCUDA::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == s_tir::attr::fragment_shape) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* shape_str = op->value.as<StringImmNode>();
    fragment_shapes[buffer] = shape_str->value;
  } else if (op->attr_key == s_tir::attr::fragment_layout) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* layout_str = op->value.as<StringImmNode>();
    fragment_layouts[buffer] = layout_str->value;
  } else if (op->attr_key == s_tir::attr::async_commit_queue_scope) {
    const IntImmNode* queue_id = op->value.as<IntImmNode>();
    TVM_FFI_ICHECK(queue_id && queue_id->value == 0)
        << "For CUDA, the index of an async queue must be 0.";
    this->VisitStmt(op->body);
    static const Op& ptx_cp_async_commit_group_op = Op::Get("tirx.ptx.cp_async_commit_group");
    auto commit_group =
        Call(PrimType::Void(), ptx_cp_async_commit_group_op, {}).as_or_throw<PrimExpr>();
    this->PrintIndent();
    this->VisitExpr(commit_group, this->stream);
    this->stream << ";\n";
    return;
  } else if (op->attr_key == s_tir::attr::async_wait_queue_scope) {
    auto wait_attrs = GetAsyncWaitAttributes(op);
    auto queue_id = wait_attrs.first.as<IntImmNode>();
    TVM_FFI_ICHECK(queue_id && queue_id->value == 0)
        << "For CUDA, the index of an async queue must be 0.";
    auto wait_cnt = wait_attrs.second;
    static const Op& ptx_cp_async_wait_group_op = Op::Get("tirx.ptx.cp_async_wait_group");
    auto wait_group =
        Call(PrimType::Void(), ptx_cp_async_wait_group_op, {wait_cnt}).as_or_throw<PrimExpr>();
    this->PrintIndent();
    this->VisitExpr(wait_group, this->stream);
    this->stream << ";\n";
    auto inner = op->body.as<AttrStmtNode>();
    TVM_FFI_ICHECK(inner);
    this->VisitStmt(inner->body);
    return;
  } else if (op->attr_key == "disable_unroll") {
    PrintIndent();
    stream << "#pragma unroll 1\n";
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "pragma_unroll") {
    PrintIndent();
    stream << "#pragma unroll\n";
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == tirx::attr::thread_extent) {
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::VisitStmt_(const AllocBufferNode* op) {
  TVM_FFI_ICHECK(op->buffer.defined());
  std::string vid = AllocVarID(op->buffer->data.get());

  this->PrintIndent();
  std::string scope = GetPtrStorageScope(op->buffer->data);
  const VarNode* buffer = op->buffer->data.get();
  PrimType dtype = op->buffer->dtype;

  if (scope.find("wmma.") == 0) {
    if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
      bool supported_wmma_input_dtype = dtype == PrimType::Float(16) || dtype == PrimType::Int(8) ||
                                        dtype == PrimType::UInt(8) || dtype == PrimType::Int(4) ||
                                        dtype == PrimType::UInt(4) || dtype == PrimType::Int(1) ||
                                        dtype == PrimType::BFloat(16);
      TVM_FFI_ICHECK(supported_wmma_input_dtype)
          << "Matrix_a and matrix_b only support half or char or unsigned char "
          << "or uint4 or int4 or int1 type for now";
    } else {
      bool supported_wmma_accumulator_dtype = dtype == PrimType::Float(16) ||
                                              dtype == PrimType::Float(32) ||
                                              dtype == PrimType::Int(32);
      TVM_FFI_ICHECK(supported_wmma_accumulator_dtype)
          << "Accumulator only support half, float and int type for now";
    }
    PrintWmmaScope(scope, dtype, buffer, stream);
  } else {
    PrintStorageScope(scope, stream);
    int align = op->buffer->data_alignment;
    auto it = op->annotations.find(tirx::attr::buffer_data_alignment);
    if (it != op->annotations.end()) {
      if (const auto* n = (*it).second.as<IntImmNode>()) {
        align = n->value;
      }
    }
    if (align > 0 && scope == "shared.dyn") {
      stream << "__align__(" << align << ") ";
    } else if (align > 0) {
      stream << "alignas(" << align << ") ";
    }
    PrintType(dtype, stream);
  }

  if (scope == "shared.dyn") {
    stream << ' ' << vid << "[];\n";
  } else {
    // Compute constant_size from buffer shape
    size_t constant_size = 1;
    for (const auto& dim : op->buffer->shape) {
      const IntImmNode* dim_imm = dim.as<IntImmNode>();
      TVM_FFI_ICHECK(dim_imm) << "Can only handle constant size stack allocation for now";
      constant_size *= dim_imm->value;
    }
    TVM_FFI_ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

    if (scope.find("wmma.") == 0) {
      constant_size = GetWmmaFragmentSize(scope, buffer, constant_size);
    }
    bool is_packed_integer_dtype =
        dtype == PrimType::Int(4) || dtype == PrimType::UInt(4) || dtype == PrimType::Int(1);
    if (is_packed_integer_dtype && scope == "shared") {
      constant_size = constant_size / (32 / dtype.bits());
    }
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer->data.get(), dtype);
  if (op->annotations.count(tirx::attr::kVolatile)) {
    MarkVolatile(op->buffer->data.get());
  }
}

void CodeGenCUDA::VisitStmt_(const EvaluateNode* op) {
  if (auto value = op->value.as<PrimExpr>(); value && is_const_int(value.value())) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call && call->op.same_as(builtin::tvm_global_barrier_kinit())) {
    PrintIndent();
    stream << "__shared__ unsigned " << vid_global_barrier_expect_ << ";\n";
    PrintIndent();
    stream << "if (threadIdx.x == 0) {\n";
    PrintIndent();
    stream << "  " << vid_global_barrier_expect_ << " = 0;\n";
    PrintIndent();
    stream << "}\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenCUDA::VisitExpr_(const RampNode* op, std::ostream& os) {
  PrimType op_ty = op->ty.as_or_throw<PrimType>();
  int lanes = op_ty.lanes();
  if (lanes <= 4) {
    PrintVecConstructor(op_ty, os);
    os << "(";
    for (int i = 0; i < lanes; i++) {
      os << "(" << PrintExpr(op->base) << ")"
         << "+(" << PrintExpr(op->stride) << "*" << i << ")";
      if (i != lanes - 1) os << ", ";
    }
    os << ")";
    return;
  }

  // Use lane-wise stores for wide vectors (e.g. fp16x8/int32x8), where CUDA
  // constructor argument layout does not match TIR vector lane layout.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(op_ty, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    std::string vbase = SSAGetID(PrintExpr(op->base), op->base.ty());
    std::string vstride = SSAGetID(PrintExpr(op->stride), op->stride.ty());
    for (int i = 0; i < lanes; ++i) {
      std::ostringstream value_temp;
      value_temp << "(" << vbase << ")+(" << vstride << "*" << i << ")";
      PrintVecElemStore(sret, op_ty, i, value_temp.str());
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenCUDA::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  PrimType op_ty = op->ty.as_or_throw<PrimType>();
  int lanes = op_ty.lanes();
  if ((op_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) && op_ty.bits() == 8 &&
      lanes == 4) {
    // make_int8x4
    const int64_t* p = as_const_int(op->value);
    TVM_FFI_ICHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    if (op_ty.MatchesCode(DLDataTypeCode::kDLUInt)) {
      os << "(uint)" << v;
    } else {
      os << "(int)" << v;
    }
    return;
  }

  if (op_ty.MatchesElementType(DLDataTypeCode::kDLFloat, 16)) {
    std::string v = PrintExpr(op->value);
    PrintVecConstructor(op_ty, os);
    os << '(';
    if (lanes <= 4) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0) os << ", ";
        os << v << ", " << v;
      }
    } else {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0) os << ", ";
        os << "__pack_half2(" << v << ", " << v << ")";
      }
    }
    os << ')';
    return;
  }

  if (op_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
    std::string v = PrintExpr(op->value);
    PrintVecConstructor(op_ty, os);
    os << '(';
    if (lanes > 4) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0) os << ", ";
        os << "__pack_nv_bfloat162(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes; ++i) {
        if (i != 0) os << ", ";
        os << v;
      }
    }
    os << ')';
    return;
  }

  if (IsFloat8(op_ty) || IsFloat4(op_ty)) {
    TVM_FFI_ICHECK(lanes == 1 || lanes == 2 || lanes == 4);
    std::string v = PrintExpr(op->value);
    // Implicit conversion from float back to fp8
    PrintType(op_ty, os);
    os << "(make_float" << lanes << "(";
    for (int i = 0; i < lanes; ++i) {
      if (i != 0) os << ", ";
      os << "static_cast<float>(" << v << ")";
    }
    os << "))";
    return;
  }

  if ((op_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) && op_ty.bits() == 4) {
    bool fail = false;
    const int64_t* p = as_const_int(op->value);
    TVM_FFI_ICHECK(p);
    int64_t v = *p & 0xF;

    if (lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (op_ty.MatchesCode(DLDataTypeCode::kDLUInt)) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) | (v << 4) | v;
      if (lanes == 8) {
        if (op_ty.MatchesCode(DLDataTypeCode::kDLUInt)) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (lanes == 16 || lanes == 32) {
        PrintVecConstructor(op_ty, os);
        os << '(';
        for (int i = 0; i < lanes / 8; ++i) {
          if (i != 0) os << ", ";
          if (op_ty.MatchesCode(DLDataTypeCode::kDLUInt)) {
            os << "(uint)" << v;
          } else {
            os << "(int)" << v;
          }
        }
        os << ')';
      } else {
        fail = true;
      }
    }

    if (!fail) {
      return;
    }
  }

  std::string v = PrintExpr(op->value);
  PrintVecConstructor(op_ty, os);
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenCUDA::VisitExpr_(const SelectNode* op, std::ostream& os) {
  PrimType op_ty = op->ty.as_or_throw<PrimType>();
  // Non-vector cases.
  if (!op_ty.IsFixedLengthVector()) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }

  // Codegen vector condition case by serializing the select op.
  TVM_FFI_ICHECK(op->false_value.ty() == op_ty && op->true_value.ty() == op_ty &&
                 op_ty.lanes() == op->condition.ty().lanes());

  std::string r_var = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(op_ty, stream);
  stream << ' ' << r_var << ";\n";
  {
    std::string c_var = SSAGetID(PrintExpr(op->condition), op_ty);
    std::string t_var = SSAGetID(PrintExpr(op->true_value), op_ty);
    std::string f_var = SSAGetID(PrintExpr(op->false_value), op_ty);

    // The condition is stored as an ushort vector.
    int lanes = op_ty.lanes();
    PrimType memory_ty = PrimType::UInt(16, lanes);

    for (int i = 0; i < lanes; ++i) {
      std::ostringstream item;
      item << "(bool(";
      PrintVecElemLoad(c_var, memory_ty, i, item);
      item << ")?";
      PrintVecElemLoad(t_var, op_ty, i, item);
      item << ':';
      PrintVecElemLoad(f_var, op_ty, i, item);
      item << ')';
      PrintVecElemStore(r_var, op_ty, i, item.str());
    }
  }
  os << r_var;
}

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenCUDA* p) {  // NOLINT(*)
  PrimType op_ty = op->ty.as_or_throw<PrimType>();
  // Type code is kBFloat
  if (op_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
    os << "__float2bfloat16_rn";
    os << '(' << std::hexfloat << op->value << 'f';
    os << "/*" << std::scientific << op->value << "*/";
    os << ')';
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (IsFloat8(op_ty) || IsFloat4(op_ty)) {
    p->PrintType(op_ty, os);
    os << '(' << std::hexfloat << op->value << 'f';
    os << "/*" << std::scientific << op->value << "*/";
    os << ')';
    return;
  }
  // Type code is kFloat
  switch (op_ty.bits()) {
    case 64: {
      std::ostringstream temp;
      if (std::isinf(op->value)) {
        if (op->value < 0) {
          temp << "-";
        }
        temp << "CUDART_INF";
        p->codegen_tags_.insert("math_constants");
      } else if (std::isnan(op->value)) {
        temp << "CUDART_NAN";
        p->codegen_tags_.insert("math_constants");
      } else {
        temp << std::fixed << std::setprecision(15) << op->value;
      }
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 32: {
      std::ostringstream temp;
      if (std::isinf(op->value)) {
        if (op->value < 0) {
          temp << "-";
        }
        temp << "CUDART_INF_F";
        p->codegen_tags_.insert("math_constants");
      } else if (std::isnan(op->value)) {
        temp << "CUDART_NAN_F";
        p->codegen_tags_.insert("math_constants");
      } else {
        temp << std::hexfloat << op->value << 'f';
        temp << "/*" << std::scientific << op->value << "*/";
      }
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << "__float2half_rn" << '(';
      FloatImm const_f32 = FloatImm(PrimType::Float(32), op->value);
      PrintConst(const_f32.get(), os, p);
      os << ')';
      break;
    }
    default:
      TVM_FFI_THROW(InternalError) << "Bad bit-width for float: " << op_ty << "\n";
  }
}

void CodeGenCUDA::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenCUDA::PrintWmmaScope(const std::string& scope, const PrimType& t,
                                 const VarNode* variable, std::ostream& os) {
  std::stringstream type;
  PrintType(t, type);
  TVM_FFI_ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  if ((t.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) && t.bits() < 8 &&
      t.lanes() == 1) {
    type.str(std::string());
    if (t.MatchesCode(DLDataTypeCode::kDLInt)) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::s4";
      } else if (t.bits() == 1) {
        type << "nvcuda::wmma::experimental::precision::b1";
      } else {
        TVM_FFI_THROW(InternalError) << "Unhandled interger type for wmma fragment!";
      }
    } else if (t.MatchesCode(DLDataTypeCode::kDLUInt)) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::u4";
      } else {
        TVM_FFI_THROW(InternalError) << "Unhandled interger type for wmma fragment!";
      }
    }
  }
  if (scope == "wmma.matrix_a") {
    codegen_tags_.insert("mma");
    std::string layout_str = fragment_layouts[variable];
    TVM_FFI_ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_a";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, " << shape_str << ", " << type.str()
       << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.matrix_b") {
    codegen_tags_.insert("mma");
    std::string layout_str = fragment_layouts[variable];
    TVM_FFI_ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_b";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, " << shape_str << ", " << type.str()
       << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.accumulator") {
    codegen_tags_.insert("mma");
    os << "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, " << shape_str << ", " << type.str()
       << ">";
  }
}

int stoi(const std::string& str) {
  try {
    return std::stoi(str);
  } catch (std::invalid_argument& e) {
    TVM_FFI_THROW(InternalError) << "Cannot convert \"" << str << "\" to int";
    throw;
  }
}

int32_t CodeGenCUDA::GetWmmaFragmentSize(const std::string& scope, const VarNode* variable,
                                         int32_t size) {
  TVM_FFI_ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  std::pair<int32_t, int32_t> dim = GetWmmaFragmentDimSize(shape_str, scope);
  if (dim.first * dim.second != 0)
    return size / dim.first / dim.second;
  else
    return 0;
}

void CodeGenCUDA::HandleVolatileLoads(const std::string& value, const BufferLoadNode* op,
                                      std::ostream& os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  //
  PrimType op_ty = op->ty.as_or_throw<PrimType>();
  if ((op_ty.MatchesElementType(DLDataTypeCode::kDLFloat, 16) ||
       op_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) &&
      IsVolatile(op->buffer->data.get())) {
    os << "(";
    PrintType(op_ty, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenCUDA::PrintVecElemLoadExpr(const PrimType& t, int i, const std::string& value,
                                       std::ostream& os) {
  int lanes = t.lanes();
  TVM_FFI_ICHECK_GT(lanes, 1);
  if (t.bits() == 8 && (t.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt))) {
    if (!(lanes == 2 || lanes == 3)) {
      if (i != 0) {
        os << "|";
      }
      os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
      return;
    }
  }

  if (t.MatchesElementType(DLDataTypeCode::kDLFloat, 16)) {
    if (i == 0) {
      PrintVecConstructor(t, os);
      os << '(';
    }
    if (i == lanes - 1) {
      os << value << ")";
    } else {
      os << value << ",";
    }
    return;
  }

  if (t.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
    if (i == 0) {
      PrintVecConstructor(t, os);
      os << '(';
    }
    if (i == lanes - 1) {
      os << value << ")";
    } else {
      os << value << ",";
    }
    return;
  }

  if (i == 0) {
    PrintVecConstructor(t, os);
    os << "(";
  }
  os << value;
  if (i != lanes - 1) {
    os << ",";
  } else {
    os << ")";
  }
  return;
}

// CUDA codegen entry point.  Generates CUDA C++ source, optionally lets a
// Python postproc hook rewrite it, and hands the source bytes off to the
// fallback-aware module factory.  The factory may JIT to PTX/cubin via
// `tvm_callback_cuda_compile` (CUDAModuleNode::JitCompileFromSource) when
// USE_CUDA=ON; on USE_CUDA=OFF builds (or when TVM_COMPILE_FORCE_FALLBACK is
// set), it returns a `CUDAFallbackModuleNode` carrying the raw source for
// later cross-compile.
ffi::Module BuildCUDA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenCUDA cg(target);
  cg.Init(output_ssa);

  ffi::Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    TVM_FFI_ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenCUDA: Can only take PrimFunc";
    auto prim_func = base_func.as_or_throw<PrimFunc>();
    CallingConv calling_conv =
        prim_func->GetAttr<CallingConv>(tvm::attr::kCallingConv, CallingConv::kDefault).value();
    TVM_FFI_ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch ||
                   calling_conv == CallingConv::kDefault)
        << "CodeGenCUDA: expect calling_conv equals CallingConv::kDeviceKernelLaunch or "
           "CallingConv::kDefault";
    functions.Set(gvar, prim_func);
  }

  for (auto [gvar, prim_func] : functions) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (auto [gvar, prim_func] : functions) {
    cg.AddFunction(gvar, prim_func);
  }

  std::string code = cg.Finish();

  if (auto f = ffi::Function::GetGlobal("tvm_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  // Hand off raw CUDA source to the fallback-aware factory.  When the real
  // CUDA runtime is registered (USE_CUDA=ON and not forced-fallback) the
  // factory invokes JitCompileFromSource via tvm_callback_cuda_compile and
  // builds a real CUDAModuleNode.  Otherwise it stores the source in a
  // CUDAFallbackModuleNode for later cross-compile.
  ffi::Map<ffi::String, ffi::String> source_map;
  return ::tvm::target::CUDAModuleCreateWithFallback(
      ffi::Bytes(code.data(), code.size()), ffi::String("cuda"), ExtractFuncInfo(mod), source_map);
}

void RegisterCudaCodegen() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.cuda", BuildCUDA);
}

TVM_FFI_STATIC_INIT_BLOCK() { RegisterCudaCodegen(); }

}  // namespace codegen
}  // namespace tvm
