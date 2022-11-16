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
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/stmt_functor.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "literal/cuda_half_t.h"
#include "ptx.h"

namespace tvm {
namespace codegen {

CodeGenCUDA::CodeGenCUDA() { restrict_keyword_ = "__restrict__"; }

void CodeGenCUDA::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  vid_global_barrier_state_ = name_supply_->FreshName(runtime::symbol::tvm_global_barrier_state);
  vid_global_barrier_expect_ = name_supply_->FreshName("__barrier_expect");
  ICHECK_EQ(vid_global_barrier_state_, runtime::symbol::tvm_global_barrier_state);
}

void CodeGenCUDA::PrintFuncPrefix() { stream << "extern \"C\" __global__ void"; }

class ThreadIdxExtractor : public tir::StmtVisitor {
 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x" || iv->thread_tag == "threadIdx.x") {
        threadIdx_x_ext = op->value;
      }
      if (iv->var->name_hint == "threadIdx.y" || iv->thread_tag == "threadIdx.y") {
        threadIdx_y_ext = op->value;
      }
      if (iv->var->name_hint == "threadIdx.z" || iv->thread_tag == "threadIdx.z") {
        threadIdx_z_ext = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

 public:
  PrimExpr threadIdx_x_ext = Integer(1);
  PrimExpr threadIdx_y_ext = Integer(1);
  PrimExpr threadIdx_z_ext = Integer(1);
};

void CodeGenCUDA::PrintExtraAttrs(const PrimFunc& f) {
  ThreadIdxExtractor extractor;
  extractor(f->body);
  arith::Analyzer analyzer;
  PrimExpr threadIdx_ext = analyzer.Simplify(extractor.threadIdx_x_ext * extractor.threadIdx_y_ext *
                                             extractor.threadIdx_z_ext);
  if (const IntImmNode* const threadIdx_ext_int = threadIdx_ext.as<IntImmNode>()) {
    if (threadIdx_ext_int->value == 1) {
      // unable to extract the number of threads per block, hence directly return
      return;
    }
    stream << " __launch_bounds__(" << threadIdx_ext_int->value << ")";
  }
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

  if (enable_bf16_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)\n";
    decl_stream << "#include <cuda_bf16.h>\n";
    decl_stream << "__device__ nv_bfloat16 max"
                << "(nv_bfloat16 a, nv_bfloat16 b)\n"
                << "{\n  return __hgt(a, b) ? a : b;\n}\n";
    decl_stream << "__device__ nv_bfloat16 min(nv_bfloat16 a, nv_bfloat16 b)\n"
                << "{\n  return __hlt(a, b) ? a : b;\n}\n";
    decl_stream << "#endif\n\n";
    decl_stream << _cuda_bfloat16_util;
  }

  if (enable_warp_shuffle_) {
    decl_stream << _cuda_warp_intrinsic_util;
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

  decl_stream << "\n#ifdef _WIN32\n";
  decl_stream << "  using uint = unsigned int;\n";
  decl_stream << "  using uchar = unsigned char;\n";
  decl_stream << "  using ushort = unsigned short;\n";
  decl_stream << "  using int64_t = long long;\n";
  decl_stream << "  using uint64_t = unsigned long long;\n";
  decl_stream << "#else\n";
  decl_stream << "  #define uint unsigned int\n";
  decl_stream << "  #define uchar unsigned char\n";
  decl_stream << "  #define ushort unsigned short\n";
  decl_stream << "  #define int64_t long long\n";
  decl_stream << "  #define uint64_t unsigned long long\n";
  decl_stream << "#endif\n";

  return CodeGenC::Finish();
}

void CodeGenCUDA::VisitStmt_(const tir::ForNode* op) {
  ICHECK(is_const_int(op->min, 0));
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] = CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenCUDA::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        enable_fp16_ = true;
        if (t.is_scalar()) {
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
          ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
          os << "uint" << lanes / 2;
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
          ICHECK_EQ(lanes % 2, 0) << "only support even lane for float type with lanes > 4";
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
    if (!fail && (t.is_scalar() || t.bits() == 16)) return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32)) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
      os << "nv_bfloat16";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
      os << "uint" << lanes / 2;
    } else {
      fail = true;
    }
    if (!fail) return;
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_vector_bool()) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
      case 1: {
        if (t.is_scalar()) {
          os << "int";
          return;
        } else if (t.lanes() == 8) {
          os << "int8_t";
          return;
        } else if (t.lanes() == 16) {
          os << "int16_t";
          return;
        } else if (t.lanes() == 32) {
          os << "int";
          return;
        } else {
          LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
        }
      }
      case 4: {
        if (t.is_scalar()) {
          os << "int";
          return;
        } else if (t.lanes() == 4) {
          os << "int16_t";
          return;
        } else if (t.lanes() == 8) {
          // directly 8 4-bit int in integer.
          os << "int";
          return;
        } else if (t.lanes() == 16) {
          os << "int2";
          return;
        } else if (t.lanes() == 32) {
          os << "int4";
          return;
        } else if (t.lanes() == 64) {
          os << "int8";
          return;
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
          os << "int";
          return;
        } else if (t.lanes() == 8) {
          enable_int8_ = true;
          os << "int2";
          return;
        } else if (t.lanes() == 16) {
          enable_int8_ = true;
          os << "int4";
          return;
        } else if (!t.is_uint() && t.is_scalar()) {
          os << "signed char";
          break;
        } else {
          os << "char";
          break;
        }
      }
      case 16: {
        if (t.is_scalar()) {
          os << "short";
        } else if (t.lanes() <= 4) {
          os << "short" << lanes;
        } else if (t.lanes() <= 8) {
          // Emit CUDA code to access int16 vector elements.
          //
          // short4 is stored as int2
          //
          // s4.x is emitted as *(short2*)(&(i2.x)).x
          // s4.y is emitted as *(short2*)(&(i2.x)).y
          // s4.z is emitted as *(short2*)(&(i2.y)).x
          // s4.w is emitted as *(short2*)(&(i2.y)).y
          //
          ICHECK_EQ(t.lanes() % 2, 0) << "only support even lane for shorT type with lanes > 4";
          os << "int" << t.lanes() / 2;
        } else {
          fail = true;
        }
        if (!fail) {
          return;
        }
        break;
      }
      case 32: {
        if (t.is_scalar()) {
          os << "int";
        } else if (t.lanes() <= 4) {
          os << "int" << t.lanes();
        } else if (t.lanes() <= 8) {
          // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
          //
          // int8 is stored as longlong4
          //
          // i8.v1 is emitted as *(int2*)(&(l4.x)).x
          // i8.v2 is emitted as *(int2*)(&(l4.x)).y
          //
          ICHECK_EQ(lanes % 2, 0) << "only support even lane for int32 type with lanes > 4";
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
        if (t.is_scalar()) {
          os << "int64_t";
        } else if (t.lanes() == 2) {
          os << "longlong2";
        } else if (t.lanes() == 3) {
          os << "longlong3";
        } else if (t.lanes() == 4) {
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
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenCUDA::PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                                   std::ostream& os) {  // NOLINT(*)
  // Delcare the result.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
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
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenCUDA::PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                   std::ostream& os) {  // NOLINT(*)
  if (t.is_scalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8 ? 16 : (t.bits() == 16 || t.bits() == 32) ? 8 : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    std::string type_name = t.is_int() ? "char" : "unsigned char";
    if (t.lanes() == 2 || t.lanes() == 3) {
      os << vec << "." << access[i % t.lanes()];
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
    }
  } else if (t.is_float16()) {
    os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
  } else if (t.is_bfloat16()) {
    os << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenCUDA::PrintVecElemStore(const std::string& vec, DataType t, int i,
                                    const std::string& value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8 ? 16 : (t.bits() == 16 || t.bits() == 32) ? 8 : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "="
             << "(" << value << ");\n";
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 4 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 4 * 8 << ");\n";
    }
  } else if (t.is_float16()) {
    stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2] << " = "
           << value << ";\n";
  } else if (t.is_bfloat16()) {
    stream << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2]
           << " = " << value << ";\n";
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
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
  ICHECK_NE(scope, "global") << "Cannot allocate global memory when targeting CUDA. You must pass "
                                "all global arrays as input instead";
  if (scope == "shared") {
    os << "__shared__ ";
  } else if (scope == "shared.dyn") {
    os << "extern __shared__ ";
  }
}

void CodeGenCUDA::VisitExpr_(const CastNode* op, std::ostream& os) {
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  // Emit simple C-style type conversion.
  if (from_ty.is_scalar()) return CodeGenC::VisitExpr_(op, os);

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
      PrintType(target_ty.element_of(), val);
      val << ")(";
      PrintVecElemLoad(src, from_ty, i, val);
      val << ")";
      PrintVecElemStore(sret, target_ty, i, val.str());
    }
  }
  os << sret;
}

void CodeGenCUDA::PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                                  bool skip_first_arg, std::ostream& os) {  // NOLINT(*)
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_vector()) {
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
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      // Load arguments.
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0) scall << ", ";
          PrintVecElemLoad(sargs[j], args[arg_begin + j].dtype(), i, scall);
        }
        scall << ")";
        PrintVecElemStore(sret, ret_dtype, i, scall.str());
      }
    }
    os << sret;
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg, os);
  }
}

void CodeGenCUDA::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (auto* ptr_op = op->op.as<OpNode>()) {
    Op call_op = GetRef<Op>(ptr_op);
    // This is only for backward compatibility with __shfl_{up/down}.
    // A macro will be used to replace *_sync calls to legacy ones.
    if (op_need_warp_shuffle_.get(call_op, false)) {
      enable_warp_shuffle_ = true;
    }
  }

  if (op->op.same_as(builtin::tvm_fill_fragment())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 6U);
    os << "nvcuda::wmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_load_matrix_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::load_matrix_sync(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_store_matrix_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
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
      LOG(FATAL) << "Invalid parameters";
    }
    os << ")";
  } else if (op->op.same_as(builtin::tvm_mma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(builtin::tvm_bmma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::bmma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(builtin::ptx_mma())) {
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
    ICHECK(op->args.size() == 13U || op->args.size() == 14U);
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    std::string A_layout = Downcast<StringImm>(op->args[1])->value;
    std::string B_layout = Downcast<StringImm>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_bias = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_bias = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_bias = this->PrintExpr(op->args[11]);
    bool saturate = Downcast<Bool>(op->args[12])->value;
    std::string bit_op = op->args.size() > 13 ? Downcast<StringImm>(op->args[13])->value : "";
    std::string asm_code =
        PrintMMAAssembly(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_bias, b_ref,
                         b_bias, c_ref, c_bias, "", "", "", bit_op, false, saturate);

    this->stream << asm_code;
  } else if (op->op.same_as(builtin::ptx_mma_sp())) {
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
    ICHECK_EQ(op->args.size(), 16U);
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    std::string A_layout = Downcast<StringImm>(op->args[1])->value;
    std::string B_layout = Downcast<StringImm>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_offset = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_offset = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_offset = this->PrintExpr(op->args[11]);
    std::string metadata = this->PrintExpr(op->args[12]);
    std::string metadata_offset = this->PrintExpr(op->args[13]);
    std::string sparse_selector = this->PrintExpr(op->args[14]);
    bool saturate = Downcast<Bool>(op->args[15])->value;
    std::string asm_code = PrintMMAAssembly(
        shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_offset, b_ref, b_offset,
        c_ref, c_offset, metadata, metadata_offset, sparse_selector, "", true, saturate);
    this->stream << asm_code;
  } else if (op->op.same_as(builtin::ptx_ldmatrix())) {
    // arg 0: whether the matrix is loaded in column major format or not.
    // arg 1: number of matrices to load.
    // arg 2: The data type in the matrix, .b16 is the only accepted data type.
    // arg 3: pointer to local buffer.
    // arg 4: The offset of the element to store in the local buffer.
    // arg 5: pointer to the shared memory buffer to load.
    // arg 6: The offset of the start element of the row to load in shared memory.
    ICHECK_EQ(op->args.size(), 7U);
    bool trans = Downcast<Bool>(op->args[0])->value;
    int num = Downcast<Integer>(op->args[1])->value;
    std::string type = Downcast<StringImm>(op->args[2])->value;
    std::string local_ptr = this->PrintExpr(op->args[3]);
    std::string local_elem_offset = this->PrintExpr(op->args[4]);
    std::string smem_ptr = this->PrintExpr(op->args[5]);
    if (trans && op->dtype.bits() == 8) {
      // Since ldmatrix assumes that a matrix element is 16 bit, it cannot properly transpose an
      // int8 matrix.
      std::string smem_stride = this->PrintExpr(op->args[6]);
      ICHECK(num == 4);
      os << "for (int i = 0; i < 16; ++i) {\n";
      os << local_ptr << "[" + local_elem_offset + " + i] = " << smem_ptr
         << "[(i % 8) / 4 * " + smem_stride + " * 16 + (threadIdx.x % 4) * 4 * " + smem_stride +
                "+ (i % 4) * " + smem_stride + " + threadIdx.x / 4 +  (i / 8) * 8];\n";
      os << "}\n";
    } else {
      std::string smem_elem_offset = this->PrintExpr(op->args[6]);
      this->stream << PrintLoadMatrixAssembly(trans, num, type, local_ptr, local_elem_offset,
                                              smem_ptr, smem_elem_offset);
    }
  } else if (op->op.same_as(builtin::mma_store())) {
    int m = Downcast<Integer>(op->args[0])->value;
    int n = Downcast<Integer>(op->args[1])->value;
    std::string dst = this->PrintExpr(op->args[2]);
    std::string src = this->PrintExpr(op->args[3]);
    std::string src_offset = this->PrintExpr(op->args[4]);
    PrimExpr stride = op->args[5];

    ICHECK(m == 16 && n == 16) << "Only m == 16 && n == 16 case supported for now";

    // Each thread in a warp holds a certain number of elements of an MMA output.
    // For example, if we compute a 16x16 tile using MMA, each thread holds 8 elements
    // in its registers. So conceptually, a warp memory is organized as a 32x8 block.
    // A map from a 16x16 tile to a 32x8 block of memory is specified by the index map below.

    // To store the 32x8 output back to a 16x16 tile in shared or global memory, we invert this map
    // to determine the output location for each 8 element.

    const auto* index_map_func =
        runtime::Registry::Get("tir.index_map.shared_16x16_to_ldmatrix_32x8_layout");
    ICHECK(index_map_func);

    auto inverse_index_map =
        IndexMap::FromFunc(2, *index_map_func).Inverse({Range(0, m), Range(0, n)});
    auto indices_16x16 = inverse_index_map->final_indices;

    // "//" and "%" in the index map are translated to FloorDiv/Mod, but the plain Div/Mod are fine.
    // FloorDiv/Mod are supposed to be lowered before they reach codegen, so manually replace them
    // to the plain ones here.
    class LowerFloorDivMod : public ExprMutator {
     public:
      PrimExpr VisitExpr_(const FloorDivNode* op) {
        return tir::Div(this->VisitExpr(op->a), this->VisitExpr(op->b));
      }
      PrimExpr VisitExpr_(const FloorModNode* op) {
        return tir::Mod(this->VisitExpr(op->a), this->VisitExpr(op->b));
      }
    };

    auto dst_ind = LowerFloorDivMod()(indices_16x16[0] * stride + indices_16x16[1]);

    var_idmap_[inverse_index_map->initial_indices[0].get()] = "threadIdx.x";
    var_idmap_[inverse_index_map->initial_indices[1].get()] = "local_id";

    os << "for (int local_id = 0; local_id < 8; ++local_id) {\n";
    os << dst << "[" + this->PrintExpr(dst_ind) + "]"
       << " = " << src << "[" << src_offset << " + local_id];\n";
    os << "}\n";

  } else if (op->op.same_as(builtin::mma_fill())) {
    std::string num_elem = this->PrintExpr(op->args[0]);
    std::string dst = this->PrintExpr(op->args[1]);
    std::string dst_offset = this->PrintExpr(op->args[2]);

    os << "for (int i = 0; i < " << num_elem << "; ++i) {\n";
    os << dst << "[" << dst_offset << " + i] = 0.0;";
    os << "}\n";
  } else if (op->op.same_as(builtin::ptx_cp_async())) {
    std::string dst = this->PrintExpr(op->args[0]);
    std::string dst_offset = this->PrintExpr(op->args[1]);
    std::string src = this->PrintExpr(op->args[2]);
    std::string src_offset = this->PrintExpr(op->args[3]);
    std::string size = this->PrintExpr(op->args[4]);
    this->stream << PrintCpAsyncAssembly(dst, dst_offset, src, src_offset, size);
  } else if (op->op.same_as(builtin::ptx_commit_group())) {
    this->stream << "__asm__ __volatile__(\"cp.async.commit_group;\");\n\n";
  } else if (op->op.same_as(builtin::ptx_wait_group())) {
    std::string N = this->PrintExpr(op->args[0]);
    this->stream << "__asm__ __volatile__(\"cp.async.wait_group " + N + ";\");\n\n";
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
  } else if (op->attr_key == tir::attr::async_commit_queue_scope) {
    const IntImmNode* queue_id = op->value.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0) << "For CUDA, the index of an async queue must be 0.";
    this->VisitStmt(op->body);
    auto commit_group = Call(DataType::Void(), builtin::ptx_commit_group(), {});
    this->VisitExpr(commit_group, this->stream);
    return;
  } else if (op->attr_key == tir::attr::async_wait_queue_scope) {
    auto wait_attrs = GetAsyncWaitAttributes(op);
    auto queue_id = wait_attrs.first.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0) << "For CUDA, the index of an async queue must be 0.";
    auto wait_cnt = wait_attrs.second;
    auto wait_group = Call(DataType::Void(), builtin::ptx_wait_group(), {wait_cnt});
    this->VisitExpr(wait_group, this->stream);
    auto inner = op->body.as<AttrStmtNode>();
    ICHECK(inner);
    this->VisitStmt(inner->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  std::string scope = GetPtrStorageScope(op->buffer_var);
  const VarNode* buffer = op->buffer_var.as<VarNode>();
  if (scope.find("wmma.") == 0) {
    if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
      ICHECK(op->dtype == DataType::Float(16) || op->dtype == DataType::Int(8) ||
             op->dtype == DataType::UInt(8) || op->dtype == DataType::Int(4) ||
             op->dtype == DataType::UInt(4) || op->dtype == DataType::Int(1) ||
             op->dtype == DataType::BFloat(16))
          << "Matrix_a and matrix_b only support half or char or unsigned char "
          << "or uint4 or int4 or int1 type for now";
    } else {
      ICHECK(op->dtype == DataType::Float(16) || op->dtype == DataType::Float(32) ||
             op->dtype == DataType::Int(32))
          << "Accumulator only support half, float and int type for now";
    }
    PrintWmmaScope(scope, op->dtype, buffer, stream);
  } else {
    PrintStorageScope(scope, stream);
    PrintType(op->dtype, stream);
  }

  if (scope == "shared.dyn") {
    stream << ' ' << vid << "[];\n";
  } else {
    size_t constant_size = op->ConstantAllocationSize();
    ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

    if (scope.find("wmma.") == 0) {
      constant_size = GetWmmaFragmentSize(scope, buffer, constant_size);
    }
    if ((op->dtype == DataType::Int(4) || op->dtype == DataType::UInt(4) ||
         op->dtype == DataType::Int(1)) &&
        scope == "shared") {
      constant_size = constant_size / (32 / op->dtype.bits());
    }
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenCUDA::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;
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
  CHECK_LE(op->lanes, 4) << "ValueError: Ramp of more than 4 lanes is not allowed.";
  os << "(make_";
  PrintType(op->dtype, os);
  os << "(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != op->lanes - 1) os << ", ";
  }
  os << "))";
}

void CodeGenCUDA::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8 && op->lanes == 4) {
    // make_int8x4
    const int64_t* p = as_const_int(op->value);
    ICHECK(p);
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
      os << "__pack_half2(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if (op->dtype.is_bfloat16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    for (int i = 0; i < op->lanes / 2; ++i) {
      if (i != 0) os << ", ";
      os << "__pack_nv_bfloat162(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 4) {
    bool fail = false;
    const int64_t* p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xF;

    if (op->lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (op->dtype.is_uint()) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) | (v << 4) | v;
      if (op->lanes == 8) {
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (op->lanes == 16 || op->lanes == 32) {
        os << "make_";
        PrintType(op->dtype, os);
        os << '(';
        for (int i = 0; i < op->lanes / 8; ++i) {
          if (i != 0) os << ", ";
          if (op->dtype.is_uint()) {
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
  os << "make_";
  PrintType(op->dtype, os);
  os << '(';
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenCUDA::VisitExpr_(const ShuffleNode* op, std::ostream& os) {
  std::vector<std::string> to_shuffle(op->vectors.size());
  for (int i = 0, e = op->vectors.size(); i < e; ++i) {
    ICHECK(op->vectors[i].dtype().lanes() == 1) << "Only scalars can be shuffled in CUDA!";
    to_shuffle[i] = PrintExpr(op->vectors[i]);
  }
  os << "make_";
  PrintType(op->dtype, os);
  os << '(';
  for (int i = 0, e = op->indices.size(); i < e; ++i) {
    const int64_t* val = as_const_int(op->indices[i]);
    ICHECK(val && *val >= 0 && (int)*val < (int)to_shuffle.size());
    if (i != 0) os << ", ";
    os << to_shuffle[*val];
  }
  os << ')';
}

void CodeGenCUDA::VisitExpr_(const SelectNode* op, std::ostream& os) {
  // Non-vector cases.
  if (!op->dtype.is_vector()) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }

  // Codegen vector condition case by serializing the select op.
  ICHECK(op->false_value->dtype == op->dtype && op->true_value->dtype == op->dtype &&
         op->dtype.lanes() == op->condition.dtype().lanes());

  std::string r_var = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(op->dtype, stream);
  stream << ' ' << r_var << ";\n";
  {
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

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenCUDA* p) {  // NOLINT(*)
  // Type code is kBFloat
  if (op->dtype.is_bfloat16()) {
    os << "__float2bfloat16_rn";
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat
  switch (op->dtype.bits()) {
    case 64:
    case 32: {
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
      os << "__float2half_rn" << '(';
      FloatImm const_f32 = FloatImm(DataType::Float(32), op->value);
      PrintConst(const_f32.get(), os, p);
      os << ')';
      break;
    }
    default:
      LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenCUDA::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenCUDA::PrintWmmaScope(const std::string& scope, DataType t, const VarNode* variable,
                                 std::ostream& os) {
  std::stringstream type;
  PrintType(t, type);
  std::string shape_str = fragment_shapes.at(variable);
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
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_a";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, " << shape_str << ", " << type.str()
       << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.matrix_b") {
    need_mma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_b";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, " << shape_str << ", " << type.str()
       << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.accumulator") {
    need_mma_h_ = true;
    os << "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, " << shape_str << ", " << type.str()
       << ">";
  }
}

int stoi(const std::string& str) {
  try {
    return std::stoi(str);
  } catch (std::invalid_argument& e) {
    LOG(FATAL) << "Cannot convert \"" << str << "\" to int";
    throw;
  }
}

int32_t CodeGenCUDA::GetWmmaFragmentSize(const std::string& scope, const VarNode* variable,
                                         int32_t size) {
  std::string shape_str = fragment_shapes.at(variable);
  size_t m, n, k;
  size_t last_pos = 0, pos = 0;
  pos = shape_str.find(", ", last_pos);
  m = tvm::codegen::stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  pos = shape_str.find(", ", last_pos);
  n = tvm::codegen::stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  k = tvm::codegen::stoi(shape_str.substr(last_pos, shape_str.length() - last_pos));
  if (scope == "wmma.matrix_a") {
    return size / m / k;
  } else if (scope == "wmma.matrix_b") {
    return size / n / k;
  } else if (scope == "wmma.accumulator") {
    return size / m / n;
  }
  return 0;
}

void CodeGenCUDA::HandleVolatileLoads(const std::string& value, const BufferLoadNode* op,
                                      std::ostream& os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  //
  if ((op->dtype.is_float16() || op->dtype.is_bfloat16()) && IsVolatile(op->buffer->data.get())) {
    os << "(";
    PrintType(op->dtype, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenCUDA::PrintVecElemLoadExpr(DataType t, int i, const std::string& value,
                                       std::ostream& os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (!(t.lanes() == 2 || t.lanes() == 3)) {
      if (i != 0) {
        os << "|";
      }
      os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
      return;
    }
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

  if (t.is_bfloat16()) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_bfloat162(" << value;
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
