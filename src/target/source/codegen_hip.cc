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
 * \file codegen_hip.cc
 */

#include "codegen_hip.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/stmt_functor.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "ptx.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Replace patterns with replacement strings.
 * \note should use std::format instead when codebase is ported to C++20.
 */
class Replacer {
 public:
  void register_rule(const std::string& pattern, const std::string& replacement) {
    _rules.emplace_back(pattern, replacement);
  }
  std::string rewrite(std::string str) {
    for (auto&& rule : _rules) {
      auto [pattern, replacement] = rule;
      size_t len = pattern.size();
      size_t new_len = replacement.size();
      size_t pos = str.find(pattern);
      while (pos != std::string::npos) {
        str = str.replace(pos, len, replacement);
        pos = str.find(pattern, pos + new_len);
      }
    }
    return str;
  }
  void empty_rules() { _rules.clear(); }

 private:
  std::vector<std::pair<std::string, std::string>> _rules;
};

CodeGenHIP::CodeGenHIP() { restrict_keyword_ = "__restrict__"; }

void CodeGenHIP::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  this->cuda_codegen_.Init(output_ssa);
}

void CodeGenHIP::PrintFuncPrefix(std::ostream& os) { os << "extern \"C\" __global__ "; }

std::string CodeGenHIP::Finish() {
  // hip must need a header file.
  decl_stream << "#include <hip/hip_runtime.h>\n";

  if (enable_fp16_) {
    decl_stream << "#include <hip/hip_fp16.h>\n";
    decl_stream << "using float16_t = _Float16;\n";
    decl_stream << "using float16x2\n";
    decl_stream << " = __attribute__((__vector_size__(2 * sizeof(float16_t)))) float16_t;\n";
    decl_stream << "using float16x4\n";
    decl_stream << " = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;\n";
    decl_stream << "using float16x8\n";
    decl_stream << " = __attribute__((__vector_size__(8 * sizeof(float16_t)))) float16_t;\n";
    decl_stream << "using float16x16\n";
    decl_stream << " = __attribute__((__vector_size__(16 * sizeof(float16_t)))) float16_t;\n";
  }

  if (need_math_constants_h_) {
    decl_stream << "#include <math_constants.h>\n";
  }

  if (need_wmma_h_) {
    decl_stream << "#include <rocwmma/rocwmma.hpp>\n";
  }
  decl_stream << "using int32x4\n";
  decl_stream << " = __attribute__((__vector_size__(4 * sizeof(int)))) int;\n";
  decl_stream << "using float32x4\n";
  decl_stream << " = __attribute__((__vector_size__(4 * sizeof(float)))) float;\n";
  decl_stream << "using float32x16\n";
  decl_stream << " = __attribute__((__vector_size__(16 * sizeof(float)))) float;\n";

  decl_stream << "#define max(a, b) (((a) > (b)) ? (a) : (b))\n";
  decl_stream << "#define min(a, b) (((a) < (b)) ? (a) : (b))\n";

  return CodeGenC::Finish();
}

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

void CodeGenHIP::PrintExtraAttrs(const PrimFunc& f, std::ostream& os) {
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
    os << " __launch_bounds__(" << threadIdx_ext_int->value << ")";
  }
}


void CodeGenHIP::VisitStmt_(const tir::ForNode* op) {
  ICHECK(is_const_int(op->min, 0));
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenHIP::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] = CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenHIP::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
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


void CodeGenHIP::PrintStorageSync(const CallNode* op) {
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

void CodeGenHIP::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  ICHECK_NE(scope, "global") << "Cannot allocate global memory when targeting CUDA. You must pass "
                                "all global arrays as input instead";
  if (scope == "shared") {
    os << "__shared__ ";
  } else if (scope == "shared.dyn") {
    os << "extern __shared__ ";
  }
}

void CodeGenHIP::VisitExpr_(const CastNode* op, std::ostream& os) {
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

void CodeGenHIP::VisitStmt_(const AttrStmtNode* op) {
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

void CodeGenHIP::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;
  CodeGenC::VisitStmt_(op);
}

void CodeGenHIP::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  int lanes = op->dtype.lanes();
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8 && lanes == 4) {
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
    PrintVecConstructor(op->dtype, os);
    os << "make_";
    //PrintType(op->dtype, os);
    os << '(';
    for (int i = 0; i < lanes / 2; ++i) {
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
    for (int i = 0; i < lanes / 2; ++i) {
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

    if (lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (op->dtype.is_uint()) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) | (v << 4) | v;
      if (lanes == 8) {
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (lanes == 16 || lanes == 32) {
        os << "make_";
        PrintType(op->dtype, os);
        os << '(';
        for (int i = 0; i < lanes / 8; ++i) {
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
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenHIP::VisitExpr_(const ShuffleNode* op, std::ostream& os) {
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

void CodeGenHIP::VisitExpr_(const SelectNode* op, std::ostream& os) {
  // Non-vector cases.
  if (!op->dtype.is_fixed_length_vector()) {
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

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenHIP* p) {  // NOLINT(*)
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
        temp << ((op->dtype.bits() == 32) ? "HIPRT_INF_F" : "HIPRT_INF");
        p->need_math_constants_h_ = true;
      } else if (std::isnan(op->value)) {
        temp << ((op->dtype.bits() == 32) ? "HIPRT_NAN_F" : "HIPRT_NAN");
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

void CodeGenHIP::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenHIP::HandleVolatileLoads(const std::string& value, const BufferLoadNode* op,
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

void CodeGenHIP::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (op->op.same_as(builtin::tvm_fill_fragment())) {
    need_wmma_h_ = true;
    ICHECK_EQ(op->args.size(), 6U);
    os << "rocwmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_load_matrix_sync())) {
    need_wmma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "rocwmma::load_matrix_sync(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_store_matrix_sync())) {
    need_wmma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "rocwmma::store_matrix_sync(";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[6], os);
    if (const StringImmNode* str = op->args[7].as<StringImmNode>()) {
      os << ", rocwmma::mem_" << str->value;
    } else {
      LOG(FATAL) << "Invalid parameters";
    }
    os << ")";
  } else if (op->op.same_as(builtin::tvm_mma_sync())) {
    need_wmma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "rocwmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(builtin::tvm_mfma())) {
    // arg 0: prefix: {otype}_16x16x16{itype}
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: float16, float32, ...
    // arg 4: B precision: float16, float32, ...
    // arg 5: C precision: float32, float64, ...
    // arg 6: A multiplicand
    // arg 7: A multiplicand index
    // arg 8: B multiplicand
    // arg 9: B multiplicand index
    // arg 10: C accumulator
    // arg 11: C accumulator index

    ICHECK(op->args.size() == 12U) << "Invalid number of arguments for tvm_mfma";
    std::string prefix = Downcast<StringImm>(op->args[0])->value;
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
    ICHECK(A_layout == "row" || B_layout == "row") << "Matrix core only support row major";
    // map for dtype -> float32x4 -> float4
    std::unordered_map<std::string, std::string> dtype_map = {
        {"int8", "char"},
        {"int32", "int"},
        {"int8x4", "int32_t"},
        {"int32x4", "int32x4"},
        {"float16", "half"},
        {"float32", "float"},
        {"float64", "double"},
        {"float16x4", "float16x4"},
        {"bfloat16x4", "bfloat16x4"},
        {"float32x4", "float32x4"},
        {"float32x16", "float32x16"}
    };
    std::string call_mfma_code = R"({
    *((({C_dytpe}*){c_ref}) + {c_bias}) = {mfma_buildin}(*((({A_dytpe}*){a_ref}) + {a_bias}),
                  *((({B_dytpe}*){b_ref}) + {b_bias}),
                  *((({C_dytpe}*){c_ref}) + {c_bias}), 0, 0, 0);
  })";
    std::string mfma_buildin = "__builtin_amdgcn_mfma_" + prefix;
    Replacer replacer;
    replacer.register_rule("{mfma_buildin}", mfma_buildin);
    replacer.register_rule("{A_dytpe}", dtype_map[A_dtype]);
    replacer.register_rule("{B_dytpe}", dtype_map[B_dtype]);
    replacer.register_rule("{C_dytpe}", dtype_map[C_dtype]);
    replacer.register_rule("{a_ref}", a_ref);
    replacer.register_rule("{a_bias}", a_bias);
    replacer.register_rule("{b_ref}", b_ref);
    replacer.register_rule("{b_bias}", b_bias);
    replacer.register_rule("{c_ref}", c_ref);
    replacer.register_rule("{c_bias}", c_bias);
    os << replacer.rewrite(call_mfma_code);
  } else if (op->op.same_as(builtin::tvm_mfma_store())) {
    int m = Downcast<Integer>(op->args[0])->value;
    int n = Downcast<Integer>(op->args[1])->value;
    std::string dst = this->PrintExpr(op->args[2]);
    std::string src = this->PrintExpr(op->args[3]);
    std::string src_offset = this->PrintExpr(op->args[4]);
    PrimExpr stride = op->args[5];

    ICHECK((m == 16 && n == 16) || (m == 32 && n==32)) << "Only m == 16 && n == 16 or m == 32 && n == 32 case supported for now";

    if(m == 16){
      // Each thread in a warp holds a certain number of elements of an MMA output.
      // For example, if we compute a 16x16 tile using MMA, each thread holds 8 elements
      // in its registers. So conceptually, a warp memory is organized as a 32x8 block.
      // A map from a 16x16 tile to a 32x8 block of memory is specified by the index map below.

      // To store the 32x8 output back to a 16x16 tile in shared or global memory, we invert this
      // map to determine the output location for each 8 element.
      const auto* index_map_func =
          runtime::Registry::Get("tir.index_map.shared_16x16_to_ldmatrix_64x4_layout");
      ICHECK(index_map_func);

      arith::Analyzer analyzer;
      auto inverse_index_map =
          IndexMap::FromFunc(2, *index_map_func).Inverse({Range(0, m), Range(0, n)}, &analyzer);
      auto indices_16x16 = inverse_index_map->final_indices;

      // "//" and "%" in the index map are translated to FloorDiv/Mod, but the plain Div/Mod are
      // fine. FloorDiv/Mod are supposed to be lowered before they reach codegen, so manually
      // replace them to the plain ones here.
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

      os << "for (int local_id = 0; local_id < 4; ++local_id) {\n";
      os << dst << "[" + this->PrintExpr(dst_ind) + "]"
         << " = (";
      this->PrintType(op->dtype, os);
      os << ")" << src << "[" << src_offset << " + local_id];\n";
      os << "}\n";
    }else{
      // Each thread in a warp holds a certain number of elements of an MMA output.
      // For example, if we compute a 16x16 tile using MMA, each thread holds 8 elements
      // in its registers. So conceptually, a warp memory is organized as a 32x8 block.
      // A map from a 16x16 tile to a 32x8 block of memory is specified by the index map below.

      // To store the 32x8 output back to a 16x16 tile in shared or global memory, we invert this
      // map to determine the output location for each 8 element.
      const auto* index_map_func =
          runtime::Registry::Get("tir.index_map.shared_32x32_to_ldmatrix_64x16_layout");
      ICHECK(index_map_func);

      arith::Analyzer analyzer;
      auto inverse_index_map =
          IndexMap::FromFunc(2, *index_map_func).Inverse({Range(0, m), Range(0, n)}, &analyzer);
      auto indices_32x32 = inverse_index_map->final_indices;

      // "//" and "%" in the index map are translated to FloorDiv/Mod, but the plain Div/Mod are
      // fine. FloorDiv/Mod are supposed to be lowered before they reach codegen, so manually
      // replace them to the plain ones here.
      class LowerFloorDivMod : public ExprMutator {
       public:
        PrimExpr VisitExpr_(const FloorDivNode* op) {
          return tir::Div(this->VisitExpr(op->a), this->VisitExpr(op->b));
        }
        PrimExpr VisitExpr_(const FloorModNode* op) {
          return tir::Mod(this->VisitExpr(op->a), this->VisitExpr(op->b));
        }
      };

      auto dst_ind = LowerFloorDivMod()(indices_32x32[0] * stride + indices_32x32[1]);

      var_idmap_[inverse_index_map->initial_indices[0].get()] = "threadIdx.x";
      var_idmap_[inverse_index_map->initial_indices[1].get()] = "local_id";

      os << "for (int local_id = 0; local_id < 16; ++local_id) {\n";
      os << dst << "[" + this->PrintExpr(dst_ind) + "]"
         << " = " << src << "[" << src_offset << " + local_id];\n";
      os << "}\n";
    }
   
  } else if (op->op.same_as(builtin::tvm_rdna_wmma())) {
    // arg 0: prefix: {otype}_16x16x16{itype}
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: float16, float32, ...
    // arg 4: B precision: float16, float32, ...
    // arg 5: C precision: float32, float64, ...
    // arg 6: A multiplicand
    // arg 7: A multiplicand index
    // arg 8: B multiplicand
    // arg 9: B multiplicand index
    // arg 10: C accumulator
    // arg 11: C accumulator index

    ICHECK(op->args.size() == 12U) << "Invalid number of arguments for tvm_mfma";
    std::string prefix = Downcast<StringImm>(op->args[0])->value;
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
    ICHECK(A_layout == "row" || B_layout == "row") << "Matrix core only support row major";
    // map for dtype -> float32x4 -> float4
    std::unordered_map<std::string, std::string> dtype_map = {
        {"int8", "char"},
        {"int32", "int"},
        {"int32x4", "int32x4"},
        {"float16", "half"},
        {"float32", "float"},
        {"float64", "double"},
        {"float16x4", "float16x4"},
        {"bfloat16x4", "bfloat16x4"},
        {"float16x8", "float16x16"},
        {"float16x16", "float16x16"},
        {"float32x4", "float32x4"},
        {"float32x16", "float32x16"}
    };
    std::string call_mfma_code = R"({
    *((({C_dytpe}*){c_ref}) + {c_bias}) = {mfma_buildin}(*((({A_dytpe}*){a_ref}) + {a_bias}),
                  *((({B_dytpe}*){b_ref}) + {b_bias}),
                  *((({C_dytpe}*){c_ref}) + {c_bias}), false);
  })";
    std::string mfma_buildin = "__builtin_amdgcn_wmma_" + prefix;
    Replacer replacer;
    replacer.register_rule("{mfma_buildin}", mfma_buildin);
    replacer.register_rule("{A_dytpe}", dtype_map[A_dtype]);
    replacer.register_rule("{B_dytpe}", dtype_map[B_dtype]);
    replacer.register_rule("{C_dytpe}", dtype_map[C_dtype]);
    replacer.register_rule("{a_ref}", a_ref);
    replacer.register_rule("{a_bias}", a_bias);
    replacer.register_rule("{b_ref}", b_ref);
    replacer.register_rule("{b_bias}", b_bias);
    replacer.register_rule("{c_ref}", c_ref);
    replacer.register_rule("{c_bias}", c_bias);
    os << replacer.rewrite(call_mfma_code);
  } else if (op->op.same_as(builtin::tvm_rdna_wmma_store())) {
    int m = Downcast<Integer>(op->args[0])->value;
    int n = Downcast<Integer>(op->args[1])->value;
    std::string dst = this->PrintExpr(op->args[2]);
    std::string src = this->PrintExpr(op->args[3]);
    std::string src_offset = this->PrintExpr(op->args[4]);
    PrimExpr stride = op->args[5];

    ICHECK((m == 16 && n == 16) || (m == 32 && n==32)) << "Only m == 16 && n == 16 or m == 32 && n == 32 case supported for now";

    if(m == 16){
      // Each thread in a warp holds a certain number of elements of an MMA output.
      // For example, if we compute a 16x16 tile using MMA, each thread holds 8 elements
      // in its registers. So conceptually, a warp memory is organized as a 32x8 block.
      // A map from a 16x16 tile to a 32x8 block of memory is specified by the index map below.

      // To store the 32x8 output back to a 16x16 tile in shared or global memory, we invert this
      // map to determine the output location for each 8 element.
      const auto* index_map_func =
          runtime::Registry::Get("tir.index_map.shared_16x16_to_ldmatrix_16x16_layout");
      ICHECK(index_map_func);

      auto inverse_index_map =
          IndexMap::FromFunc(2, *index_map_func);
      auto indices_16x16 = inverse_index_map->final_indices;

      // "//" and "%" in the index map are translated to FloorDiv/Mod, but the plain Div/Mod are
      // fine. FloorDiv/Mod are supposed to be lowered before they reach codegen, so manually
      // replace them to the plain ones here.
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
         << " = " << src << "[" << src_offset << " + local_id * 2];\n";
      os << "}\n";
    }
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenHIP::VisitStmt_(const AllocateNode* op) {
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

void CodeGenHIP::PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                                 bool skip_first_arg, std::ostream& os) {  // NOLINT(*)
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_fixed_length_vector()) {
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

void CodeGenHIP::PrintWmmaScope(const std::string& scope, DataType t, const VarNode* variable,
                                std::ostream& os) {
  std::stringstream type;
  PrintType(t, type);
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  if ((t.is_int() || t.is_uint()) && t.bits() < 8 && t.lanes() == 1) {
    type.str(std::string());
    if (t.is_int()) {
      if (t.bits() == 4) {
        type << "rocwmma::experimental::precision::s4";
      } else if (t.bits() == 1) {
        type << "rocwmma::experimental::precision::b1";
      } else {
        LOG(FATAL) << "Unhandled interger type for wmma fragment!";
      }
    } else if (t.is_uint()) {
      if (t.bits() == 4) {
        type << "rocwmma::experimental::precision::u4";
      } else {
        LOG(FATAL) << "Unhandled interger type for wmma fragment!";
      }
    }
  }
  if (scope == "wmma.matrix_a") {
    need_wmma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_a";
    os << "rocwmma::fragment<rocwmma::matrix_a, " << shape_str << ", " << type.str()
       << ", rocwmma::" << layout_str << ">";
  } else if (scope == "wmma.matrix_b") {
    need_wmma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_b";
    os << "rocwmma::fragment<rocwmma::matrix_b, " << shape_str << ", " << type.str()
       << ", rocwmma::" << layout_str << ">";
  } else if (scope == "wmma.accumulator") {
    need_wmma_h_ = true;
    os << "rocwmma::fragment<rocwmma::accumulator, " << shape_str << ", " << type.str() << ">";
  }
}

int32_t CodeGenHIP::GetWmmaFragmentSize(const std::string& scope, const VarNode* variable,
                                        int32_t size) {
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;

  auto stoi = [](const std::string& str) {
    try {
      return std::stoi(str);
    } catch (std::invalid_argument& e) {
      LOG(FATAL) << "Cannot convert \"" << str << "\" to int";
      throw;
    }
  };

  std::string shape_str = fragment_shapes.at(variable);
  size_t m, n, k;
  size_t last_pos = 0, pos = 0;
  pos = shape_str.find(", ", last_pos);
  m = stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  pos = shape_str.find(", ", last_pos);
  n = stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  k = stoi(shape_str.substr(last_pos, shape_str.length() - last_pos));
  if (scope == "wmma.matrix_a") {
    return size / m / k;
  } else if (scope == "wmma.matrix_b") {
    return size / n / k;
  } else if (scope == "wmma.accumulator") {
    return size / m / n;
  }
  return 0;
}

void CodeGenHIP::PrintVecElemLoadExpr(DataType t, int i, const std::string& value,
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
