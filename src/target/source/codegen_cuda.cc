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

#include <tvm/runtime/registry.h>

#include <cmath>
#include <utility>
#include <vector>
#include <string>
#include "literal/cuda_half_t.h"
#include "codegen_cuda.h"

namespace tvm {
namespace codegen {

CodeGenCUDA::CodeGenCUDA() {
  restrict_keyword_ = "__restrict__";
}

void CodeGenCUDA::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  vid_global_barrier_state_ = GetUniqueName(runtime::symbol::tvm_global_barrier_state);
  vid_global_barrier_expect_ = GetUniqueName("__barrier_expect");
  CHECK_EQ(vid_global_barrier_state_, runtime::symbol::tvm_global_barrier_state);
}


void CodeGenCUDA::PrintFuncPrefix() {
  stream << "extern \"C\" __global__ void";
}

std::string CodeGenCUDA::Finish() {
  if (enable_fp16_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)\n";
    decl_stream << "#include <cuda_fp16.h>\n";
    decl_stream << "__device__ half max"
                << "(half a, half b)\n"
                << "{\n  return __hgt(__half(a), __half(b)) ? a : b;\n}\n";
    decl_stream << "__device__ half min(half a, half b)\n"
                << "{\n  return __hlt(__half(a), __half(b)) ? a : b;\n}\n";
    decl_stream << "#else\n";
    decl_stream << _cuda_half_t_def;
    decl_stream << "#endif\n\n";
    decl_stream << _cuda_half_util;
  }

  if (enable_int8_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)\n";
    decl_stream << "#include <sm_61_intrinsics.h>\n";
    decl_stream << "#endif\n";
  }

  if (need_math_constants_h_) {
    decl_stream << "#include <math_constants.h>\n";
  }

  if (need_mma_h_) {
    decl_stream << "#include <mma.h>\n";
  }

  return CodeGenC::Finish();
}

void CodeGenCUDA::VisitStmt_(const tir::ForNode* op) {
  CHECK(is_const_int(op->min, 0));
  if (op->for_type == tir::ForType::Unrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::BindThreadIndex(const IterVar& iv) {
  CHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenCUDA::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1)
        << "do not yet support vector types";
    os << "void*"; return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        enable_fp16_ = true;
        if (lanes == 1) {
          os << "half";
        } else if (lanes <= 8) {
          // Emit CUDA code to access fp16 vector elements.
          //
          // half4 is stored as uint2
          //
          // h4.x is emitted as *(half2*)(&(u2.x)).x
          // h4.y is emitted as *(half2*)(&(u2.x)).y
          // h4.z is emitted as *(half2*)(&(u2.y)).x
          // h4.w is emitted as *(half2*)(&(u2.y)).y
          //
          CHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
          os << "uint" << lanes / 2;
        } else {
          fail = true;
        }
        break;
      case 32: os << "float"; break;
      case 64: os << "double"; break;
      default: fail = true; break;
    }
    if (!fail && (lanes == 1 || t.bits() == 16)) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  } else if (t == DataType::Bool()) {
    os << "bool"; return;
  } else if (t.is_vector_bool()) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n; return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      if (t.lanes() != 1) {
        os << "u";
      } else {
        os << "unsigned ";
      }
    }
    switch (t.bits()) {
      case 1: {
        if (t.lanes() == 1) {
          os << "int"; return;
        } else if (t.lanes() == 8) {
          os << "int8_t"; return;
        } else if (t.lanes() == 16) {
          os << "int16_t"; return;
        } else if (t.lanes() == 32) {
          os << "int"; return;
        } else {
          LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
        }
      }
      case 4: {
        if (t.lanes() == 1) {
          os << "int"; return;
        } else if (t.lanes() == 4) {
          os << "int16_t"; return;
        } else if (t.lanes() == 8) {
          // directly 8 4-bit int in integer.
          os << "int"; return;
        } else if (t.lanes() == 16) {
          os << "int2"; return;
        } else if (t.lanes() == 32) {
          os << "int4"; return;
        } else if (t.lanes() == 64) {
          os << "int8"; return;
        } else {
          LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
        }
      }
      case 8: {
        if (t.lanes() == 4) {
          // directly 4 8 bit int in integer.
          enable_int8_ = true;

          // We use int for int8x4 instead of char4 because using char4 is
          // likely to produce extra instructions to pack four int8 elements
          // into 32-bit data.
          os << "int"; return;
        } else if (t.lanes() == 8) {
          enable_int8_ = true;
          os << "int2"; return;
        } else if (t.lanes() == 16) {
          enable_int8_ = true;
          os << "int4"; return;
        } else if (!t.is_uint() && t.lanes() == 1) {
          os << "signed char"; break;
        } else {
          os << "char"; break;
        }
      }
      case 16: os << "short"; break;
      case 32: os << "int"; break;
      case 64: {
        if (sizeof(long) != 8) { // NOLINT(*)
          if (t.lanes() == 1) {
            os << "long long"; break;
          } else if (t.lanes() == 2) {
            os << "longlong"; break;
          } else {
            // No longlong3, longlong4
            LOG(FATAL) << "Cannot convert type " << t << " to CUDA type on a L32 platform";
            break;
          }
        } else {
          os << "long"; break;
        }
      }
      default: fail = true; break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenCUDA::PrintVecBinaryOp(
    const std::string& op, DataType t,
    PrimExpr lhs, PrimExpr rhs, std::ostream& os) {  // NOLINT(*)
  // Delcare the result.
  std::string sret = GetUniqueName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  {
    EnterScopeRAII scope(this);

    // Unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());

    for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  os << sret;
}

void CodeGenCUDA::PrintVecElemLoad(
    const std::string& vec, DataType t, int i, std::ostream& os) {  // NOLINT(*)
  static const char access[] = {'x', 'y', 'z', 'w'};
  CHECK(i >= 0 && i < (t.is_float16() ? 8 : 4));
  if ((t.is_int()) && t.bits() == 8) {
    os << "((char)(" << vec << " >> " << i * 8 << "))";
  } else if ((t.is_uint()) && t.bits() == 8) {
    os << "((unsigned char)(" << vec << " >> " << i * 8 << "))";
  } else if (t.is_float16()) {
    os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
       << access[i % 2];
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenCUDA::PrintVecElemStore(
    const std::string& vec, DataType t, int i, const std::string& value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  CHECK(i >= 0 && i < (t.is_float16() ? 8 : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    stream << vec << "=";
    // Do not read the first undef lane.
    if (i != 0) {
      stream << vec << " & ~(0x000000ff << " << i * 8 << ") |";
    }
    stream << "(" << value << " << " << i * 8 << ");\n";
  } else if (t.is_float16()) {
    stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
           << access[i % 2] << " = " << value << ";\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenCUDA::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  } else if (sync == "global") {
    if (!need_global_barrier_) {
      need_global_barrier_ = true;
      this->decl_stream << "extern \"C\" __device__ unsigned "
                        << vid_global_barrier_state_ << ";\n";
    }
    // global synchronizer
    std::string is_load = PrintExpr(op->args[1]);
    std::string num_blocks = PrintExpr(op->args[2]);
    this->PrintIndent();
    // In theory only threadfence is needed
    // but we observed problems with only threadfence
    this->stream <<"__threadfence_system();\n";
    this->PrintIndent();
    this->stream <<"if (" << is_load << ") {\n";
    int wb = this->BeginScope();
    this->PrintIndent();
    this->stream << "atomicAdd(&" << vid_global_barrier_state_ << ", 1);\n";
    this->PrintIndent();
    std::string ptr = GetUniqueName("pf");
    this->stream << "volatile unsigned* "
                 << ptr << " = &" << vid_global_barrier_state_<< ";\n";
    this->PrintIndent();
    this->stream << vid_global_barrier_expect_ << " += " << num_blocks << ";\n";
    this->PrintIndent();
    this->stream <<"while (" << ptr << "[0] < " << vid_global_barrier_expect_ << ");\n";
    this->EndScope(wb);
    this->PrintIndent();
    this->stream <<"}\n";
    this->PrintIndent();
    this->stream <<"__syncthreads();\n";
  }
}

void CodeGenCUDA::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
  CHECK_NE(scope, "global");
  if (scope == "shared") {
    os << "__shared__";
  }
}

void CodeGenCUDA::VisitExpr_(const CastNode* op, std::ostream& os) {
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  CHECK_EQ(target_ty.lanes(), from_ty.lanes());

  // Emit simple C-style type conversion.
  if (from_ty.is_scalar())
    return CodeGenC::VisitExpr_(op, os);

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = GetUniqueName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  {
    EnterScopeRAII scope(this);
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
      std::ostringstream val;
      val << "(";
      PrintType(target_ty.element_of(), val);
      val << ")(";
      PrintVecElemLoad(src, from_ty, i, val);
      val << ")";
      PrintVecElemStore(sret, target_ty, i, val.str());
    }
  }
  os << sret;
}

void CodeGenCUDA::VisitExpr_(const CallNode *op, std::ostream& os) {
  if (op->is_intrinsic(intrinsic::tvm_fill_fragment)) {
    need_mma_h_ = true;
    CHECK_EQ(op->args.size(), 6U);
    os << "nvcuda::wmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ")";
  } else if (op->is_intrinsic(intrinsic::tvm_load_matrix_sync)) {
    need_mma_h_ = true;
    CHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::load_matrix_sync(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    os << ")";
  } else if (op->is_intrinsic(intrinsic::tvm_store_matrix_sync)) {
    need_mma_h_ = true;
    CHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::store_matrix_sync(";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[6], os);
    if (const StringImmNode *str = op->args[7].as<StringImmNode>()) {
      os << ", nvcuda::wmma::mem_" << str->value;
    } else {
      LOG(FATAL) << "Invalid parameters";
    }
    os << ")";
  } else if (op->is_intrinsic(intrinsic::tvm_mma_sync)) {
    need_mma_h_ = true;
    CHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", ": ")");
    }
  } else if (op->is_intrinsic(intrinsic::tvm_bmma_sync)) {
    need_mma_h_ = true;
    CHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::bmma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", ": ")");
    }
  } else if (op->call_type == CallNode::PureExtern && op->dtype.is_vector()) {
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
    std::string sret = GetUniqueName("_");
    this->PrintIndent();
    this->PrintType(op->dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      EnterScopeRAII scope(this);

      // Load arguments.
      std::vector<std::string> sargs;
      for (size_t i = 0; i < op->args.size(); ++i) {
        std::string val = SSAGetID(PrintExpr(op->args[i]), op->args[i].dtype());
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
      for (int i = 0; i < op->dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << op->name << "(";
        for (size_t j = 0; j < op->args.size(); ++j) {
          if (j > 0)
            scall << ", ";
          PrintVecElemLoad(sargs[j], op->args[j].dtype(), i, scall);
        }
        scall << ")";
        PrintVecElemStore(sret, op->dtype, i, scall.str());
      }
    }
    os << sret;
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCUDA::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::fragment_shape) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* shape_str = op->value.as<StringImmNode>();
    fragment_shapes[buffer] = shape_str->value;
  } else if (op->attr_key == tir::attr::fragment_layout) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* layout_str = op->value.as<StringImmNode>();
    fragment_layouts[buffer] = layout_str->value;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::VisitStmt_(const AllocateNode* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  int32_t constant_size = op->constant_allocation_size();
  CHECK_GT(constant_size, 0)
    << "Can only handle constant size stack allocation for now";
  const VarNode* buffer = op->buffer_var.as<VarNode>();
  std::string scope = alloc_storage_scope_.at(buffer);
  if (scope.find("wmma.") == 0) {
    if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
      CHECK(op->dtype == DataType::Float(16) ||
            op->dtype == DataType::Int(8) ||
            op->dtype == DataType::UInt(8) ||
            op->dtype == DataType::Int(4) ||
            op->dtype == DataType::UInt(4) ||
            op->dtype == DataType::Int(1))
        << "Matrix_a and matrix_b only support half or char or unsigned char "
        << "or uint4 or int4 or int1 type for now";
    } else {
      CHECK(op->dtype == DataType::Float(16) ||
            op->dtype == DataType::Float(32) ||
            op->dtype == DataType::Int(32))
        << "Accumulator only support half, float and int type for now";
    }
    constant_size = GetWmmaFragmentSize(scope, buffer, constant_size);
    PrintWmmaScope(scope, op->dtype, buffer, stream);
  } else {
    PrintStorageScope(scope, stream);
    stream << ' ';
    PrintType(op->dtype, stream);
  }
  if ((op->dtype == DataType::Int(4) ||
        op->dtype == DataType::UInt(4) ||
        op->dtype == DataType::Int(1)) && scope == "shared") {
    constant_size = constant_size / (32 / op->dtype.bits());
  }
  stream << ' '<< vid << '['
          << constant_size << "];\n";

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenCUDA::VisitStmt_(const EvaluateNode *op) {
  if (is_const(op->value)) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call && call->is_intrinsic(intrinsic::tvm_global_barrier_kinit)) {
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
  os << "((make_int" << op->lanes << ")(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")" << "+(" << PrintExpr(op->stride) << "*" << i <<")";
    if (i != op->lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenCUDA::VisitExpr_(const BroadcastNode* op, std::ostream& os) {   // NOLINT(*)
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8 && op->lanes == 4) {
    // make_int8x4
    const int64_t *p = as_const_int(op->value);
    CHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    if (op->dtype.is_uint()) {
      os << "(uint)" << v;
    } else {
      os << "(int)" << v;
    }
    return;
  }

  if (op->dtype.is_float16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    for (int i = 0; i < op->lanes / 2; ++i) {
      if (i != 0) os << ", ";
      os << "__pack_half2(" << v << ", " << v <<  ")";
    }
    os << ')';
    return;
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(op->dtype, os);
  os << '(';
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenCUDA::VisitExpr_(const ShuffleNode* op, std::ostream &os) {
  std::vector<std::string> to_shuffle(op->vectors.size());
  for (int i = 0, e = op->vectors.size(); i < e; ++i) {
    CHECK(op->vectors[i].dtype().lanes() == 1) << "Only scalars can be shuffled in CUDA!";
    to_shuffle[i] = PrintExpr(op->vectors[i]);
  }
  os << "make_";
  PrintType(op->dtype, os);
  os << '(';
  for (int i = 0, e = op->indices.size(); i < e; ++i) {
    const int64_t *val = as_const_int(op->indices[i]);
    CHECK(val && *val >= 0 && (int) *val < (int) to_shuffle.size());
    if (i != 0) os << ", ";
    os << to_shuffle[*val];
  }
  os << ')';
}

void CodeGenCUDA::VisitExpr_(const SelectNode* op, std::ostream &os) {
  // Non-vector cases.
  if (!op->dtype.is_vector()) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }

  // Codegen vector condition case by serializing the select op.
  CHECK(op->false_value->dtype == op->dtype &&
        op->true_value->dtype == op->dtype &&
        op->dtype.lanes() == op->condition.dtype().lanes());

  std::string r_var = GetUniqueName("_");
  this->PrintIndent();
  this->PrintType(op->dtype, stream);
  stream << ' ' << r_var << ";\n";
  {
    EnterScopeRAII scope(this);

    std::string c_var = SSAGetID(PrintExpr(op->condition), op->dtype);
    std::string t_var = SSAGetID(PrintExpr(op->true_value), op->dtype);
    std::string f_var = SSAGetID(PrintExpr(op->false_value), op->dtype);

    // The condition is stored as an ushort vector.
    int lanes = op->dtype.lanes();
    DataType memory_ty(DataType::TypeCode::kUInt, 16, lanes);

    for (int i = 0; i < lanes; ++i) {
      std::ostringstream item;
      item << "(bool(";
      PrintVecElemLoad(c_var, memory_ty, i, item);
      item << ")?";
      PrintVecElemLoad(t_var, op->dtype, i, item);
      item << ':';
      PrintVecElemLoad(f_var, op->dtype, i, item);
      item << ')';
      PrintVecElemStore(r_var, op->dtype, i, item.str());
    }
  }
  os << r_var;
}

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenCUDA* p) { // NOLINT(*)
  switch (op->dtype.bits()) {
    case 64: case 32: {
      std::ostringstream temp;
      if (std::isinf(op->value)) {
        if (op->value < 0) {
          temp << "-";
        }
        temp << ((op->dtype.bits() == 32) ? "CUDART_INF_F" : "CUDART_INF");
        p->need_math_constants_h_ = true;
      } else if (std::isnan(op->value)) {
        temp << ((op->dtype.bits() == 32) ? "CUDART_NAN_F" : "CUDART_NAN");
        p->need_math_constants_h_ = true;
      } else {
        temp << std::scientific << op->value;
        if (op->dtype.bits() == 32) temp << 'f';
      }
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << "__float2half_rn";
      os << '(' << std::scientific << op->value << 'f' << ')';
      break;
    }
    default: LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}


void CodeGenCUDA::VisitExpr_(const FloatImmNode *op, std::ostream& os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenCUDA::PrintWmmaScope(const std::string &scope, DataType t,
    const VarNode* variable, std::ostream &os) {
  std::stringstream type;
  PrintType(t, type);
  std::string shape_str = fragment_shapes[variable];
  if ((t.is_int() || t.is_uint()) && t.bits() < 8 && t.lanes() == 1) {
    type.str(std::string());
    if (t.is_int()) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::s4";
      } else if (t.bits() == 1) {
        type << "nvcuda::wmma::experimental::precision::b1";
      } else {
        LOG(FATAL) << "Unhandled interger type for wmma fragment!";
      }
    } else if (t.is_uint()) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::u4";
      } else {
        LOG(FATAL) << "Unhandled interger type for wmma fragment!";
      }
    }
  }
  if (scope == "wmma.matrix_a") {
    need_mma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, "
      << shape_str << ", " << type.str() << ", nvcuda::wmma::" << layout_str <<">";
  } else if (scope == "wmma.matrix_b") {
    need_mma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, "
       << shape_str << ", " << type.str() << ", nvcuda::wmma::" << layout_str <<">";
  } else if (scope == "wmma.accumulator") {
    need_mma_h_ = true;
    os << "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, "
       << shape_str << ", "<< type.str() << ">";
  }
}

int32_t CodeGenCUDA::GetWmmaFragmentSize(const std::string &scope,
                                         const VarNode* variable, int32_t size) {
  std::string shape_str = fragment_shapes[variable];
  size_t m, n, k;
  size_t last_pos = 0, pos = 0;
  pos = shape_str.find(", ", last_pos);
  m = std::stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  pos = shape_str.find(", ", last_pos);
  n = std::stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  k = std::stoi(shape_str.substr(last_pos, shape_str.length() - last_pos));
  if (scope == "wmma.matrix_a") {
    return size / m / k;
  } else if (scope == "wmma.matrix_b") {
    return size / n / k;
  } else if (scope == "wmma.accumulator") {
    return size / m / n;
  }
  return 0;
}

void CodeGenCUDA::HandleVolatileLoads(const std::string& value,
                                      const LoadNode* op, std::ostream& os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  //
  if (op->dtype.is_float16() && IsVolatile(op->buffer_var.get())) {
    os << "(";
    PrintType(op->dtype, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenCUDA::PrintVecElemLoadExpr(
    DataType t, int i, const std::string& value, std::ostream& os) {
  CHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (i != 0) {
      os << "|";
    }
    os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
    return;
  }

  if (t.is_float16()) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_half2(" << value;
    } else {
      os << "," << value << ")";
      if (i != t.lanes() - 1) {
        os << ",";
      } else {
        os << ")";
      }
    }
    return;
  }

  if (i == 0) {
    os << "make_";
    PrintType(t, os);
    os << "(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << ")";
  }
  return;
}

}  // namespace codegen
}  // namespace tvm
