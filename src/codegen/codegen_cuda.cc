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
 *  Copyright (c) 2017 by Contributors
 * \file codegen_cuda.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>
#include <cmath>
#include <vector>
#include <string>
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

void CodeGenCUDA::AddFunction(LoweredFunc f) {
  this->stream << "extern \"C\" __global__ ";
  CodeGenC::AddFunction(f);
}

std::string CodeGenCUDA::Finish() {
  if (enable_fp16_) {
    decl_stream << "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)\n";
    decl_stream << "#include <cuda_fp16.h>\n";
    decl_stream << "__device__ half max" \
                    "(const half a, const half b)\n"
                    "{\n  return __hgt(__half(a), __half(b)) ? a : b;\n}\n";
    decl_stream << "__device__ half min(const half a, const half b)\n"
                    "{\n  return __hlt(__half(a), __half(b)) ? a : b;\n}\n";
    decl_stream << "__device__ half operator+" \
                    "(const volatile __half &a,  const volatile __half &b)\n"
                    "{\n  return __hadd(a, b);\n}\n";
    decl_stream << "__device__ half operator<=" \
                   "(const volatile __half &a,  const volatile __half &b)\n"
                    "{\n  return __hlt(a, b);\n}\n";
    decl_stream << "__device__ half operator*" \
                    "(const volatile __half &a,  const volatile __half &b)\n"
                    "{\n  return __hmul(a, b);\n}\n";
    decl_stream << "#else\n";
    decl_stream << "typedef unsigned short uint16_t;\n";
    decl_stream << "typedef unsigned char uint8_t;\n";
    decl_stream << "typedef int int32_t;\n";
    decl_stream << "typedef unsigned long long uint64_t;\n";
    decl_stream << "typedef unsigned int uint32_t;\n";
    decl_stream << "#define TVM_FORCE_INLINE inline __attribute__((always_inline))\n";
    decl_stream << "#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__\n";
    decl_stream << "#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))\n";
    decl_stream << "#define TVM_HALF_OPERATOR(RTYPE, OP)                              \\\n"
                   "  TVM_XINLINE RTYPE operator OP (half a, half b) {                \\\n"
                   "    return RTYPE(float(a) OP float(b));                           \\\n"
                   "  }                                                               \\\n"
                   "  template<typename T>                                            \\\n"
                   "  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \\\n"
                   "    return RTYPE(float(a) OP float(b));                           \\\n"
                   "  }                                                               \\\n"
                   "  template<typename T>                                            \\\n"
                   "  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \\\n"
                   "    return RTYPE(float(a) OP float(b));                           \\\n"
                   "  }\n"
                   "\n";
    decl_stream << "#define TVM_HALF_ASSIGNOP(AOP, OP)                                \\\n"
                   "  template<typename T>                                            \\\n"
                   "  TVM_XINLINE half operator AOP (const T& a) {                    \\\n"
                   "    return *this = half(float(*this) OP float(a));                \\\n"
                   "  }                                                               \\\n"
                   "  template<typename T>                                            \\\n"
                   "  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \\\n"
                   "    return *this = half(float(*this) OP float(a));                \\\n"
                   "  }\n\n";
    decl_stream << "class TVM_ALIGNED(2) half {\n"
                   " public:\n"
                   "  uint16_t half_;\n"
                   "\n"
                   "  static TVM_XINLINE half Binary(uint16_t value) {\n"
                   "    half res;\n"
                   "    res.half_ = value;\n"
                   "    return res;\n"
                   "  }\n"
                   "\n"
                   "  TVM_XINLINE half() {}\n"
                   "\n"
                   "  TVM_XINLINE half(const float& value) { constructor(value); }\n"
                   "  TVM_XINLINE explicit half(const double& value) { constructor(value); }\n"
                   "  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }\n"
                   "  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }\n"
                   "  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }\n"
                   "  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }\n"
                   "  TVM_XINLINE explicit half(const int64_t& value) { constructor(value); }\n"
                   "  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }\n"
                   "\n"
                   "  TVM_XINLINE operator float() const {                          \\\n"
                   "    return float(half2float(half_));                            \\\n"
                   "  }                                                             \\\n"
                   "  TVM_XINLINE operator float() const volatile {                 \\\n"
                   "    return float(half2float(half_));                            \\\n"
                   "  }\n\n"
                   "\n"
                   "  TVM_HALF_ASSIGNOP(+=, +)\n"
                   "  TVM_HALF_ASSIGNOP(-=, -)\n"
                   "  TVM_HALF_ASSIGNOP(*=, *)\n"
                   "  TVM_HALF_ASSIGNOP(/=, /)\n"
                   "\n"
                   "  TVM_XINLINE half operator+() {\n"
                   "    return *this;\n"
                   "  }\n"
                   "\n"
                   "  TVM_XINLINE half operator-() {\n"
                   "    return half(-float(*this));  \n"
                   "  }\n"
                   "\n"
                   "  TVM_XINLINE half operator=(const half& a) {\n"
                   "    half_ = a.half_;\n"
                   "    return a;\n"
                   "  }\n"
                   "\n"
                   "  template<typename T>\n"
                   "  TVM_XINLINE half operator=(const T& a) {\n"
                   "    return *this = half(a);  \n"
                   "  }\n"
                   "\n"
                   "  TVM_XINLINE half operator=(const half& a) volatile {\n"
                   "    half_ = a.half_;\n"
                   "    return a;\n"
                   "  }\n"
                   "\n"
                   "  template<typename T>\n"
                   "  TVM_XINLINE half operator=(const T& a) volatile {\n"
                   "    return *this = half(a);  \n"
                   "  }\n"
                   "\n"
                   " private:\n"
                   "  union Bits {\n"
                   "    float f;\n"
                   "    int32_t si;\n"
                   "    uint32_t ui;\n"
                   "  };\n"
                   "\n"
                   "  static int const fp16FractionBits = 10;\n"
                   "  static int const fp32FractionBits = 23;\n"
                   "  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);  // == 0x7fffff\n"
                   "  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;         // == 0x800000\n"
                   "  static int const shift = fp32FractionBits - fp16FractionBits;       // == 13\n"
                   "  static int const shiftSign = 16;\n"
                   "  // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)\n"
                   "  static int32_t const expAdjust = 127 - 15;\n"
                   "\n"
                   "  static int32_t const infN = 0x7F800000;  // flt32 infinity\n"
                   "  static int32_t const maxN = 0x477FFFFF;  // max flt32 that's a flt16 normal after >> by shift\n"
                   "  static int32_t const minN = 0x38800000;  // min flt16 normal as a flt32\n"
                   "  static int32_t const maxZ = 0x33000000;  // max fp32 number that's still rounded to zero in fp16\n"
                   "  static int32_t const signN = 0x80000000;  // flt32 sign bit\n"
                   "\n"
                   "  static int32_t const infC = infN >> shift;\n"
                   "  static int32_t const nanN = (infC + 1) << shift;  // minimum flt16 nan as a flt32\n"
                   "  static int32_t const maxC = maxN >> shift;\n"
                   "  static int32_t const minC = minN >> shift;\n"
                   "  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit\n"
                   "\n"
                   "  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN\n"
                   "  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))\n"
                   "\n"
                   "  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted\n"
                   "  static int32_t const norC = 0x00400;  // min flt32 normal down shifted\n"
                   "\n"
                   "  static int32_t const maxD = infC - maxC - 1;\n"
                   "  static int32_t const minD = minC - subC - 1;\n"
                   "\n"
                   "  TVM_XINLINE uint16_t float2half(const float& value) const {\n"
                   "    Bits v;\n"
                   "    v.f = value;\n"
                   "    uint32_t sign = v.si & signN;    // grab sign bit\n"
                   "    v.si ^= sign;                    // clear sign bit from v\n"
                   "    sign >>= shiftSign;              // logical shift sign to fp16 position\n"
                   "\n"
                   "    if (v.si <= maxZ) {\n"
                   "      // Handle eventual zeros here to ensure vshift will not exceed 32 below.\n"
                   "      v.ui = 0;\n"
                   "    } else if (v.si < minN) {\n"
                   "      // Handle denorms\n"
                   "      uint32_t exp32 = v.ui >> fp32FractionBits;\n"
                   "      int32_t exp16 = exp32 - expAdjust;\n"
                   "      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.\n"
                   "      // Smaller (so negative) exp16 values should result in greater right shifts.\n"
                   "      uint32_t vshift = 1 - exp16;\n"
                   "      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);\n"
                   "      v.ui = significand >> vshift;\n"
                   "    } else if (v.si <= maxN) {\n"
                   "      // Handle norms\n"
                   "      v.ui -= expAdjust << fp32FractionBits;\n"
                   "    } else if (v.si <= infN) {\n"
                   "      v.si = infN;\n"
                   "    } else if (v.si < nanN) {\n"
                   "      v.si = nanN;\n"
                   "    }\n"
                   "\n"
                   "    v.ui >>= shift;\n"
                   "    return sign | (v.ui & 0x7fff);\n"
                   "  }\n"
                   "\n"
                   "  // Same as above routine, except for addition of volatile keyword\n"
                   "  TVM_XINLINE uint16_t float2half(const volatile float& value) const volatile {  \n"
                   "    Bits v;\n"
                   "    v.f = value;\n"
                   "    uint32_t sign = v.si & signN;    // grab sign bit\n"
                   "    v.si ^= sign;                    // clear sign bit from v\n"
                   "    sign >>= shiftSign;              // logical shift sign to fp16 position\n"
                   "\n"
                   "    if (v.si <= maxZ) {\n"
                   "      // Handle eventual zeros here to ensure vshift will not exceed 32 below.\n"
                   "      v.ui = 0;\n"
                   "    } else if (v.si < minN) {\n"
                   "      // Handle denorms\n"
                   "      uint32_t exp32 = v.ui >> fp32FractionBits;\n"
                   "      int32_t exp16 = exp32 - expAdjust;\n"
                   "      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.\n"
                   "      // Smaller (so negative) exp16 values should result in greater right shifts.\n"
                   "      uint32_t vshift = 1 - exp16;\n"
                   "      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);\n"
                   "      v.ui = significand >> vshift;\n"
                   "    } else if (v.si <= maxN) {\n"
                   "      // Handle norms\n"
                   "      v.ui -= expAdjust << fp32FractionBits;\n"
                   "    } else if (v.si <= infN) {\n"
                   "      v.si = infN;\n"
                   "    } else if (v.si < nanN) {\n"
                   "      v.si = nanN;\n"
                   "    }\n"
                   "\n"
                   "    v.ui >>= shift;\n"
                   "    return sign | (v.ui & 0x7fff);\n"
                   "  }\n"
                   "\n"
                   "  TVM_XINLINE float half2float(const uint16_t& value) const {\n"
                   "    Bits v;\n"
                   "    v.ui = value;\n"
                   "    int32_t sign = v.si & signC;\n"
                   "    v.si ^= sign;\n"
                   "    sign <<= shiftSign;\n"
                   "    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);\n"
                   "    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);\n"
                   "    Bits s;\n"
                   "    s.si = mulC;\n"
                   "    s.f *= v.si;\n"
                   "    int32_t mask = -(norC > v.si);\n"
                   "    v.si <<= shift;\n"
                   "    v.si ^= (s.si ^ v.si) & mask;\n"
                   "    v.si |= sign;\n"
                   "    return v.f;\n"
                   "  }\n"
                   "\n"
                   "  TVM_XINLINE float half2float(const volatile uint16_t& value) const volatile {  \n"
                   "    Bits v;\n"
                   "    v.ui = value;\n"
                   "    int32_t sign = v.si & signC;\n"
                   "    v.si ^= sign;\n"
                   "    sign <<= shiftSign;\n"
                   "    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);\n"
                   "    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);\n"
                   "    Bits s;\n"
                   "    s.si = mulC;\n"
                   "    s.f *= v.si;\n"
                   "    int32_t mask = -(norC > v.si);\n"
                   "    v.si <<= shift;\n"
                   "    v.si ^= (s.si ^ v.si) & mask;\n"
                   "    v.si |= sign;\n"
                   "    return v.f;\n"
                   "  }\n"
                   "\n"
                   "  template<typename T>\n"
                   "  TVM_XINLINE void constructor(const T& value) {\n"
                   "    half_ = float2half(float(value));  \n"
                   "  }\n"
                   "};\n"
                   "\n"

                   "TVM_HALF_OPERATOR(half, +)\n"
                   "TVM_HALF_OPERATOR(half, -)\n"
                   "TVM_HALF_OPERATOR(half, *)\n"
                   "TVM_HALF_OPERATOR(half, /)\n"
                   "TVM_HALF_OPERATOR(bool, >)\n"
                   "TVM_HALF_OPERATOR(bool, <)\n"
                   "TVM_HALF_OPERATOR(bool, >=)\n"
                   "TVM_HALF_OPERATOR(bool, <=)\n";
    decl_stream << "#endif\n\n";
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

void CodeGenCUDA::VisitStmt_(const ir::For* op) {
  CHECK(is_const_int(op->min, 0));
  if (op->for_type == ir::ForType::Unrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::BindThreadIndex(const IterVar& iv) {
  CHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, UInt(32), iv->var.type());
}

void CodeGenCUDA::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
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
          CHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
          os << "float" << lanes / 2;
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
  } else if (t == Bool()) {
    os << "bool"; return;
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      if (t.lanes() != 1) {
        os << "u";
      } else {
        os << "unsigned ";
      }
    }
    switch (t.bits()) {
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
          }
        } else {
          os << "long"; break;
        }
      }
      case 1: os << "int"; break;
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
    const std::string&op, Type t,
    Expr lhs, Expr rhs, std::ostream& os) {  // NOLINT(*)
  // unpacking operations.
  int lanes = t.lanes();

  {
    // The assignment below introduces side-effect, and the resulting value cannot
    // be reused across multiple expression, thus a new scope is needed
    int vec_scope = BeginScope();

    // default: unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.type());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.type());
    std::string sret = GetUniqueName("_");
    {
      // delcare type.
      this->PrintIndent();
      this->PrintType(t, stream);
      stream << ' ' << sret << ";\n";
    }
    for (int i = 0; i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.type(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.type(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.type(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.type(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
    os << sret;
    EndScope(vec_scope);
  }
}

void CodeGenCUDA::PrintVecElemLoad(
    const std::string& vec, Type t, int i, std::ostream& os) {  // NOLINT(*)
  static const char access[] = {'x', 'y', 'z', 'w'};
  CHECK(i >= 0 && i < 4);
  if (t.is_int() && t.bits() == 8) {
    os << "(0x000000ff & (" << vec << " >> " << i * 8 << "))";
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenCUDA::PrintVecElemStore(
    const std::string& vec, Type t, int i, const std::string& value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  CHECK(i >= 0 && i < 4);
  if (t.is_int() && t.bits() == 8) {
    stream << vec << "=" << vec << " & ~(0x000000ff << " << i * 8 << ") | ("
        << value << " << " << i * 8 << ");\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenCUDA::PrintStorageSync(const Call* op) {
  const std::string& sync = op->args[0].as<StringImm>()->value;
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

void CodeGenCUDA::VisitExpr_(const Call *op, std::ostream& os) {
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
    if (const StringImm *str = op->args[7].as<StringImm>()) {
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
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenCUDA::VisitStmt_(const AttrStmt* op) {
  if (op->attr_key == attr::fragment_shape) {
    const Variable* buffer = op->node.as<Variable>();
    const StringImm* shape_str = op->value.as<StringImm>();
    fragment_shapes[buffer] = shape_str->value;
  } else if (op->attr_key == attr::fragment_layout) {
    const Variable* buffer = op->node.as<Variable>();
    const StringImm* layout_str = op->value.as<StringImm>();
    fragment_layouts[buffer] = layout_str->value;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenCUDA::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  if (op->new_expr.defined()) {
    // Prefer global static allocation for the program
    CHECK_EQ(op->free_function, "nop");
    std::string new_data = PrintExpr(op->new_expr);
    this->PrintIndent();
    PrintType(op->type, stream);
    stream << "* "<< vid << '=' << new_data << ";\n";
  } else {
    this->PrintIndent();
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";
    const Variable* buffer = op->buffer_var.as<Variable>();
    std::string scope = alloc_storage_scope_.at(buffer);
    if (scope.find("wmma.") == 0) {
      if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
        CHECK(op->type == Float(16) || op->type == Int(8) || op->type == UInt(8))
          << "Matrix_a and matrix_b only support half or char or unsigned char type for now";
      } else {
        CHECK(op->type == Float(16) || op->type == Float(32) || op->type == Int(32))
          << "Accumulator only support half, float and int type for now";
      }
      constant_size = GetWmmaFragmentSize(scope, buffer, constant_size);
      PrintWmmaScope(scope, op->type, buffer, stream);
    } else {
      PrintStorageScope(scope, stream);
      stream << ' ';
      PrintType(op->type, stream);
    }
    stream << ' '<< vid << '['
           << constant_size << "];\n";
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeGenCUDA::VisitStmt_(const Evaluate *op) {
  if (is_const(op->value)) return;
  const Call* call = op->value.as<Call>();
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

void CodeGenCUDA::VisitExpr_(const Ramp* op, std::ostream& os) {
  os << "((make_int" << op->lanes << ")(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")" << "+(" << PrintExpr(op->stride) << "*" << i <<")";
    if (i != op->lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenCUDA::VisitExpr_(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
  if (op->type.is_int() && op->type.bits() == 8 && op->lanes == 4) {
    // make_int8x4
    const int64_t *p = as_const_int(op->value);
    CHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    os << "(int)" << v;
    return;
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(op->type, os);
  os << '(';
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenCUDA::VisitExpr_(const Shuffle* op, std::ostream &os) {
  std::vector<std::string> to_shuffle(op->vectors.size());
  for (int i = 0, e = op->vectors.size(); i < e; ++i) {
    CHECK(op->vectors[i].type().lanes() == 1) << "Only scalars can be shuffled in CUDA!";
    to_shuffle[i] = PrintExpr(op->vectors[i]);
  }
  os << "make_";
  PrintType(op->type, os);
  os << '(';
  for (int i = 0, e = op->indices.size(); i < e; ++i) {
    const int64_t *val = as_const_int(op->indices[i]);
    CHECK(val && *val >= 0 && (int) *val < (int) to_shuffle.size());
    if (i != 0) os << ", ";
    os << to_shuffle[*val];
  }
  os << ')';
}

inline void PrintConst(const FloatImm* op, std::ostream& os, CodeGenCUDA* p) { // NOLINT(*)
  switch (op->type.bits()) {
    case 64: case 32: {
      std::ostringstream temp;
      if (std::isinf(op->value)) {
        if (op->value < 0) {
          temp << "-";
        }
        temp << ((op->type.bits() == 32) ? "CUDART_INF_F" : "CUDART_INF");
        p->need_math_constants_h_ = true;
      } else if (std::isnan(op->value)) {
        temp << ((op->type.bits() == 32) ? "CUDART_NAN_F" : "CUDART_NAN");
        p->need_math_constants_h_ = true;
      } else {
        temp << std::scientific << op->value;
        if (op->type.bits() == 32) temp << 'f';
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
    default: LOG(FATAL) << "Bad bit-width for float: " << op->type << "\n";
  }
}


void CodeGenCUDA::VisitExpr_(const FloatImm *op, std::ostream& os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenCUDA::PrintWmmaScope(const std::string &scope, Type t,
    const Variable* variable, std::ostream &os) {
  std::stringstream type;
  PrintType(t, type);
  std::string shape_str = fragment_shapes[variable];
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
                                         const Variable* variable, int32_t size) {
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

}  // namespace codegen
}  // namespace tvm
