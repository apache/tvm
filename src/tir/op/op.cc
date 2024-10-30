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
 * \file tir/op/op.cc
 *
 *  Common operator definitions for ops in tir/op.h
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <cmath>
// Centralized header for constant folders.
#include "../../arith/const_fold.h"
#include "../../arith/scalable_expression.h"
#include "../../target/datatype/registry.h"

namespace tvm {

using namespace tir;

// macro to register an unary op
#define TVM_TIR_REGISTER_PURE_UNARY_OP(OpName)                             \
  TVM_TIR_REGISTER_OP(OpName).set_num_inputs(1).set_attr<TCallEffectKind>( \
      "TCallEffectKind", Integer(CallEffectKind::kPure))

// macro to register an binary op
#define TVM_TIR_REGISTER_PURE_BINARY_OP(OpName)                            \
  TVM_TIR_REGISTER_OP(OpName).set_num_inputs(2).set_attr<TCallEffectKind>( \
      "TCallEffectKind", Integer(CallEffectKind::kPure))

runtime::DataType GetRuntimeDataType(const Type& type) {
  if (auto* n = type.as<PrimTypeNode>()) {
    return n->dtype;
  } else if (type.as<PointerTypeNode>()) {
    return DataType::Handle();
  } else if (IsVoidType(type)) {
    return DataType::Void();
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding runtime::DataType";
  }
}

Type GetType(const PrimExpr& expr) {
  // TODO(tqchen): add recursive type inference for Call here
  // once we introduced the corresponding fields to the IR.
  if (auto* ptr = expr.as<tir::VarNode>()) {
    // If Var has a more refined type annotation,
    // return the type anotation
    if (ptr->type_annotation.defined()) {
      return ptr->type_annotation;
    }
  }

  if (auto* access = expr.as<tir::CallNode>()) {
    if (access->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK(access->args.size()) << "Builtin tvm_access_ptr() may not have empty arguments";
      auto type_annotation = Downcast<Call>(access->args[0]);
      static auto builtin_op = Op::Get("tir.type_annotation");
      ICHECK(type_annotation->op.same_as(builtin_op))
          << "Expected the first argument of builtin tvm_access_ptr() "
          << "to be a type annotation, but found " << type_annotation->op;
      return PointerType(PrimType(type_annotation->dtype));
    }
  }

  if (auto* address_of = expr.as<tir::CallNode>()) {
    if (address_of->op.same_as(builtin::address_of())) {
      ICHECK_EQ(address_of->args.size(), 1)
          << "Builtin address_of() expects a single argument, but received arguments "
          << address_of->args;
      auto* address = address_of->args[0].as<BufferLoadNode>();
      ICHECK(address)
          << "Builtin address_of() expects the argument to be a BufferLoad, but received argument "
          << address_of->args[0];

      return PointerType(PrimType(address->dtype));
    }
  }
  // Default: return the type indicated by the dtype.
  runtime::DataType dtype = expr.dtype();
  return GetTypeFromRuntimeDataType(dtype);
}

Type GetTypeFromRuntimeDataType(const DataType& dtype) {
  if (dtype.is_void()) {
    return VoidType();
  }
  return PrimType(dtype);
}

// LargeUIntImm
PrimExpr LargeUIntImm(DataType t, int64_t low, int64_t high, Span span) {
  return tir::Call(
      t, tir::builtin::large_uint_imm(),
      {make_const(DataType::UInt(32), low, span), make_const(DataType::UInt(32), high, span)},
      span);
}

// Q-multiplication
PrimExpr q_multiply_shift(PrimExpr x, PrimExpr y, PrimExpr q, PrimExpr s, Span span) {
  return tir::Call(DataType::Int(32, x.dtype().lanes()), tir::builtin::q_multiply_shift(),
                   {x, y, q, s}, span);
}

void BroadcastToMatchLanes(PrimExpr& op_a, PrimExpr& op_b) {  // NOLINT(*)
  DataType dtype_a = op_a.dtype();
  DataType dtype_b = op_b.dtype();

  if (!dtype_a.is_scalable_or_fixed_length_vector() &&
      dtype_b.is_scalable_or_fixed_length_vector()) {
    if (dtype_b.is_scalable_vector()) {
      op_a = tir::Broadcast(
          op_a, tir::Mul(dtype_b.vscale_factor(), Call(DataType::Int(32), builtin::vscale(), {})));
    } else {
      op_a = tir::Broadcast(op_a, dtype_b.lanes());
    }
  }
}

// The public function with a quick checking path.
void BinaryOpMatchTypes(PrimExpr& lhs, PrimExpr& rhs, Span span) {  // NOLINT(*)
  CHECK(lhs.defined()) << "ValueError: `lhs` is null in the binary operator";
  CHECK(rhs.defined()) << "ValueError: `rhs` is null in the binary operator";
  if (lhs.dtype() == rhs.dtype()) return;

  BroadcastToMatchLanes(lhs, rhs);
  BroadcastToMatchLanes(rhs, lhs);

  DataType ltype = lhs.dtype();
  DataType rtype = rhs.dtype();

  ICHECK(ltype.is_scalable_vector() == rtype.is_scalable_vector())
      << "Can't match scalable and fixed length vectors";

  bool lanes_match = false;

  if (ltype.is_scalable_vector()) {
    lanes_match = ltype.vscale_factor() == rtype.vscale_factor();
  } else {
    lanes_match = ltype.lanes() == rtype.lanes();
  }

  ICHECK(lanes_match) << "Cannot match type " << ltype << " vs " << rtype;
  if (lhs.dtype() == rhs.dtype()) return;

  ltype = lhs.dtype();
  rtype = rhs.dtype();
  // We keep dtypes conversion to be relatively consistent to reduce the amount code generated by
  // operators. This can be helpful for users to find potential type conversion problems. The
  // following are exceptions:
  if (ltype.is_float() && rtype.is_float()) {
    // Given two dissimilar floats, cast the lower bit version to the higher bit version.
    // E.g. fp16 + fp32 --> fp32 + fp32
    if (ltype.bits() < rtype.bits()) {
      lhs = cast(rtype, lhs);
    } else {
      rhs = cast(ltype, rhs);
    }
  } else if (!ltype.is_float() &&
             (rtype.is_float() || datatype::Registry::Global()->GetTypeRegistered(rtype.code()))) {
    // Cast int->float when the other operand is a float
    lhs = cast(rtype, lhs);
  } else if ((ltype.is_float() || datatype::Registry::Global()->GetTypeRegistered(ltype.code())) &&
             !rtype.is_float()) {
    // Cast int->float when the other operand is a float
    rhs = cast(ltype, rhs);
  } else if (!ltype.is_bfloat16() &&
             (rtype.is_bfloat16() ||
              datatype::Registry::Global()->GetTypeRegistered(rtype.code()))) {
    // Cast int->bfloat16 when the other operand is a bfloat16
    lhs = cast(rtype, lhs);
  } else if ((ltype.is_bfloat16() ||
              datatype::Registry::Global()->GetTypeRegistered(ltype.code())) &&
             !rtype.is_bfloat16()) {
    // Cast int->bfloat16 when the other operand is a bfloat16
    rhs = cast(ltype, rhs);
  } else if (!ltype.is_float8() && rtype.is_float8()) {
    // Cast int->float8 for lhs when rhs is a float8
    lhs = cast(rtype, lhs);
  } else if (ltype.is_float8() && !rtype.is_float8()) {
    // Cast int->float8 for rhs when lhs is a float8
    rhs = cast(ltype, rhs);
  } else if ((ltype.is_int() && rtype.is_int()) || (ltype.is_uint() && rtype.is_uint())) {
    // Promote int to higher bits e.g. int8 + int16 --> int16 + int16
    if (ltype.bits() < rtype.bits()) {
      lhs = cast(rtype, lhs);
    } else {
      rhs = cast(ltype, rhs);
    }
  } else if ((ltype.is_int() && rtype.is_uint()) || (ltype.is_uint() && rtype.is_int())) {
    // Handle mixing signed and unsigned integers
    if (ltype.bits() < rtype.bits()) {
      lhs = cast(rtype, lhs);
    } else if (ltype.bits() > rtype.bits()) {
      rhs = cast(ltype, rhs);
    } else {
      // The width of signed and unsigned integers is same.
      if (ltype.is_uint()) {
        rhs = cast(ltype, rhs);
      } else {
        lhs = cast(rtype, lhs);
      }
    }
  } else {
    LOG(INFO) << lhs << " " << rhs;
    LOG(FATAL) << "Cannot match type " << ltype << " vs " << rtype;
  }
}

PrimExpr ret(PrimExpr value, Span span) {
  CHECK(value.defined());
  return tir::Call(value.dtype(), tir::builtin::ret(), {value}, span);
}

TVM_REGISTER_GLOBAL("tir.ret").set_body_typed(ret);

// maximum and min limits
PrimExpr max_value(const DataType& dtype, Span span) {
  using namespace tir;
  ICHECK_EQ(dtype.lanes(), 1);
  if (dtype.is_int()) {
    if (dtype.bits() == 64) {
      return IntImm(dtype, std::numeric_limits<int64_t>::max(), span);
    } else if (dtype.bits() < 64) {
      int64_t val = 1;
      val = (val << (dtype.bits() - 1)) - 1;
      return IntImm(dtype, val, span);
    }
  } else if (dtype.is_uint()) {
    if (dtype.bits() == 64) {
      return make_const(dtype, std::numeric_limits<uint64_t>::max(), span);
    } else if (dtype.bits() < 64) {
      uint64_t val = 1;
      val = (val << static_cast<uint64_t>(dtype.bits())) - 1;
      return IntImm(dtype, static_cast<int64_t>(val), span);
    }
  } else if (dtype.is_float()) {
    if (dtype.bits() == 64) {
      return FloatImm(dtype, std::numeric_limits<double>::max(), span);
    } else if (dtype.bits() == 32) {
      return FloatImm(dtype, std::numeric_limits<float>::max(), span);
    } else if (dtype.bits() == 16) {
      return FloatImm(dtype, 65504.0, span);
    }
  } else if (dtype.is_bfloat16()) {
    return FloatImm(dtype, std::numeric_limits<float>::max(), span);
  } else if (dtype.is_float8()) {
    // according to https://arxiv.org/pdf/2209.05433.pdf
    if (dtype.code() == DataType::TypeCode::kE5M2Float) {
      return FloatImm(dtype, 57344.0, span);
    } else if (dtype.code() == DataType::TypeCode::kE4M3Float) {
      return FloatImm(dtype, 448.0, span);
    }
  }
  LOG(FATAL) << "Cannot decide max_value for type" << dtype;
}

PrimExpr min_value(const DataType& dtype, Span span) {
  using namespace tir;
  ICHECK_EQ(dtype.lanes(), 1);
  if (datatype::Registry::Global()->GetTypeRegistered(dtype.code())) {
    // TODO(tkonolige): need to convert all registered min functions to use the span.
    auto f = datatype::GetMinFunc(dtype.code());
    ICHECK(f) << "No minimum function registered for custom dtype " << (unsigned int)dtype.code();
    // TODO(@hypercubestart) Document this change (and others associated with the overflowing
    // floatimm min bug)
    return (*f)(dtype.bits());
  } else if (dtype.is_int()) {
    if (dtype.bits() == 64) {
      return IntImm(dtype, std::numeric_limits<int64_t>::lowest(), span);
    } else if (dtype.bits() < 64) {
      int64_t val = 1;
      val = -(val << (dtype.bits() - 1));
      return IntImm(dtype, val, span);
    }
  } else if (dtype.is_uint()) {
    return IntImm(dtype, 0, span);
  } else if (dtype.is_float()) {
    if (dtype.bits() == 64) {
      return FloatImm(dtype, std::numeric_limits<double>::lowest(), span);
    } else if (dtype.bits() == 32) {
      return FloatImm(dtype, std::numeric_limits<float>::lowest(), span);
    } else if (dtype.bits() == 16) {
      return FloatImm(dtype, -65504.0, span);
    }
  } else if (dtype.is_bfloat16()) {
    return FloatImm(dtype, std::numeric_limits<float>::lowest(), span);
  } else if (dtype.is_float8()) {
    // according to https://arxiv.org/pdf/2209.05433.pdf
    if (dtype.code() == DataType::TypeCode::kE5M2Float) {
      return FloatImm(dtype, -57344.0, span);
    } else if (dtype.code() == DataType::TypeCode::kE4M3Float) {
      return FloatImm(dtype, -448.0, span);
    }
  }
  LOG(FATAL) << "Cannot decide min_value for type" << dtype;
}

// infinity
PrimExpr infinity(const DataType& dtype, Span span) {
  using namespace tir;
  ICHECK_EQ(dtype.lanes(), 1);
  if (dtype.is_float()) {
    if (dtype.bits() == 64) {
      return FloatImm(dtype, std::numeric_limits<double>::infinity(), span);
    } else if (dtype.bits() == 32 || dtype.bits() == 16) {
      return FloatImm(dtype, std::numeric_limits<float>::infinity(), span);
    }
  }
  LOG(FATAL) << "Cannot decide infinity for type " << dtype;
}

namespace tir {
template <typename ValueType>
inline bool ConstPowerHelper(ValueType val, int* shift) {
  if (val <= 0) return false;
  shift[0] = 0;
  while (val != 0) {
    if (val & 1) {
      return (val == 1);
    }
    ++shift[0];
    val = val >> 1;
  }
  return true;
}

bool is_const_power_of_two_integer(const PrimExpr& x, int* shift) {
  if (const auto* op = x.as<tir::IntImmNode>()) {
    return ConstPowerHelper(op->value, shift);
  } else {
    return false;
  }
}
}  // namespace tir

PrimExpr cast(const DataType& t, PrimExpr value, Span span) {
  using tir::FloatImmNode;
  if (value.dtype() == t) return value;
  // const fold IntImm as they are used in index computations
  if (t.is_scalar()) {
    if (const IntImmNode* op = value.as<IntImmNode>()) {
      return make_const(t, op->value, op->span);
    } else if (const FloatImmNode* op = value.as<FloatImmNode>()) {
      return make_const(t, op->value, op->span);
    }
    ICHECK(!value.dtype().is_handle()) << "Can't cast a handle to other types.";
    return tir::Cast(t, value, span);
  } else {
    DataType vtype = t.element_of();
    if (!value.dtype().is_scalable_or_fixed_length_vector()) {
      // manually unroll cast
      if (value.dtype() != vtype) {
        if (const IntImmNode* op = value.as<IntImmNode>()) {
          value = make_const(vtype, op->value, op->span);
        } else if (const FloatImmNode* op = value.as<FloatImmNode>()) {
          value = make_const(vtype, op->value, op->span);
        } else {
          value = tir::Cast(vtype, value, span);
        }
      }
      if (t.is_scalable_vector()) {
        return tir::Broadcast(
            value, tir::Mul(t.vscale_factor(), Call(DataType::Int(32), builtin::vscale(), {})),
            span);
      } else {
        return tir::Broadcast(value, t.lanes(), span);
      }
    } else { /* value is a vector */
      ICHECK(value.dtype().is_scalable_vector() == t.is_scalable_vector());

      bool lanes_match = false;
      if (value.dtype().is_scalable_vector()) {
        lanes_match = value.dtype().vscale_factor() == t.vscale_factor();
      } else {
        lanes_match = value.dtype().lanes() == t.lanes();
      }
      ICHECK(lanes_match);
      if (const auto* broadcast = value.as<tir::BroadcastNode>()) {
        return tir::Broadcast(cast(vtype, broadcast->value, span), broadcast->lanes, span);
      } else if (const auto* ramp = value.as<tir::RampNode>()) {
        if (t.is_int() || t.is_uint()) {
          // only cast to index data type can be folded to ramp
          return tir::Ramp(cast(vtype, ramp->base, span), cast(vtype, ramp->stride, span),
                           ramp->lanes, span);
        }
      }
      return tir::Cast(t, value, span);
    }
  }
}

// reinterpret
PrimExpr reinterpret(const DataType& t, PrimExpr value, Span span) {
  if (value.dtype() == t) return value;
  if (!t.is_scalable_vector() && !value.dtype().is_scalable_vector()) {
    ICHECK(value.dtype().bits() * value.dtype().lanes() == t.bits() * t.lanes())
        << "Bitcast requires size match " << t << " vs " << value.dtype();
  }
  return tir::Call(t, tir::builtin::reinterpret(), {value}, span);
}

// operator+
PrimExpr operator+(PrimExpr a, PrimExpr b) { return add(a, b); }

PrimExpr add(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::Add>(a, b)) return ret.value();
  return tir::Add(a, b, span);
}

// negation
PrimExpr operator-(PrimExpr a) { return neg(a); }

PrimExpr neg(PrimExpr a, Span span) {
  using tir::FloatImmNode;
  using tir::IntImmNode;
  const IntImmNode* pa = a.as<IntImmNode>();
  const FloatImmNode* fa = a.as<FloatImmNode>();
  if (pa) return IntImm(a.dtype(), -pa->value, span);
  if (fa) return FloatImm(a.dtype(), -fa->value, span);
  return make_zero(a.dtype(), span) - a;
}

PrimExpr operator-(PrimExpr a, PrimExpr b) { return sub(a, b); }

PrimExpr sub(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::Sub>(a, b)) return ret.value();
  return tir::Sub(a, b, span);
}

PrimExpr operator*(PrimExpr a, PrimExpr b) { return mul(a, b); }
PrimExpr mul(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::Mul>(a, b)) return ret.value();
  return tir::Mul(a, b, span);
}

PrimExpr div(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::Div>(a, b)) return ret.value();
  return tir::Div(a, b, span);
}

PrimExpr truncdiv(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  ICHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  return div(a, b, span);
}

PrimExpr truncmod(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::Mod>(a, b)) return ret.value();
  return tir::Mod(a, b, span);
}

PrimExpr operator/(PrimExpr a, PrimExpr b) { return div(a, b); }

PrimExpr operator%(PrimExpr a, PrimExpr b) { return truncmod(a, b); }

// TODO(tqchen): switch to floordiv
PrimExpr indexdiv(PrimExpr a, PrimExpr b, Span span) { return floordiv(a, b, span); }

PrimExpr shapediv(PrimExpr a, PrimExpr b, Span span) { return ceildiv(a, b, span); }

PrimExpr indexmod(PrimExpr a, PrimExpr b, Span span) { return floormod(a, b, span); }

PrimExpr floordiv(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  ICHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::FloorDiv>(a, b)) return ret.value();
  return tir::FloorDiv(a, b, span);
}

PrimExpr ceildiv(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  ICHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::FloorDiv>(a + b - 1, b)) return ret.value();
  return tir::FloorDiv(a + b - 1, b, span);
}

PrimExpr floormod(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  ICHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::FloorMod>(a, b)) return ret.value();
  return tir::FloorMod(a, b, span);
}

PrimExpr min(PrimExpr a, PrimExpr b, Span span) {
  // inf-aware simplificaiton
  using arith::is_neg_inf;
  using arith::is_pos_inf;
  if (is_pos_inf(a)) return b;
  if (is_neg_inf(a)) return a;
  if (is_pos_inf(b)) return a;
  if (is_neg_inf(b)) return b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::Min>(a, b)) return ret.value();
  return tir::Min(a, b, span);
}

PrimExpr max(PrimExpr a, PrimExpr b, Span span) {
  // inf-aware simplificaiton
  using arith::is_neg_inf;
  using arith::is_pos_inf;
  if (is_pos_inf(a)) return a;
  if (is_neg_inf(a)) return b;
  if (is_pos_inf(b)) return b;
  if (is_neg_inf(b)) return a;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::Max>(a, b)) return ret.value();
  return tir::Max(a, b, span);
}

// if_then_else
PrimExpr if_then_else(PrimExpr cond, PrimExpr true_value, PrimExpr false_value, Span span) {
  ICHECK(cond.dtype() == DataType::Bool(1))
      << "if_then_else only accept the condition to be boolean type.";
  BinaryOpMatchTypes(true_value, false_value, span);
  if (const IntImmNode* op = cond.as<IntImmNode>()) {
    if (op->value != 0) {
      return true_value;
    } else {
      return false_value;
    }
  }

  return tir::Call(true_value.dtype(), tir::builtin::if_then_else(),
                   {cond, true_value, false_value}, span);
}

// likely
PrimExpr likely(PrimExpr cond, Span span) {
  if (is_const_int(cond)) return cond;
  return tir::Call(cond.dtype(), tir::builtin::likely(), {cond}, span);
}

// operator>
PrimExpr operator>(PrimExpr a, PrimExpr b) { return greater(a, b); }
PrimExpr greater(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::GT>(a, b)) return ret.value();
  return tir::GT(a, b, span);
}

PrimExpr operator>=(PrimExpr a, PrimExpr b) { return greater_equal(a, b); }
PrimExpr greater_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::GE>(a, b)) return ret.value();
  return tir::GE(a, b, span);
}

PrimExpr operator<(PrimExpr a, PrimExpr b) { return less(a, b); }
PrimExpr less(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::LT>(a, b)) return ret.value();
  return tir::LT(a, b, span);
}

PrimExpr operator<=(PrimExpr a, PrimExpr b) { return less_equal(a, b); }
PrimExpr less_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::LE>(a, b)) return ret.value();
  return tir::LE(a, b, span);
}

PrimExpr operator==(PrimExpr a, PrimExpr b) { return equal(a, b); }
PrimExpr equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::EQ>(a, b)) return ret.value();
  if (arith::IsVScaleCall(a) && arith::IsVScaleCall(b)) return true;
  return tir::EQ(a, b, span);
}

PrimExpr operator!=(PrimExpr a, PrimExpr b) { return not_equal(a, b); }
PrimExpr not_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tir::NE>(a, b)) return ret.value();
  return tir::NE(a, b, span);
}

namespace {
void type_check_boolean_args(const PrimExpr& arg, const char* op) {
  ICHECK(arg.dtype().is_bool()) << "Expected boolean argument for " << op << ", but received "
                                << arg << " of type " << arg.dtype();
}
void type_check_boolean_args(const PrimExpr& lhs, const PrimExpr& rhs, const char* op) {
  ICHECK(lhs.dtype().is_bool()) << "Expected boolean argument as LHS of " << op << ", but received "
                                << lhs << " of type " << lhs.dtype();
  ICHECK(rhs.dtype().is_bool()) << "Expected boolean argument as RHS of " << op << ", but received "
                                << rhs << " of type " << rhs.dtype();
}

void type_check_integer_args(const PrimExpr& arg, const char* op) {
  ICHECK(arg.dtype().is_int() || arg.dtype().is_uint())
      << "Expected integer argument for " << op << ", but received " << arg << " of type "
      << arg.dtype();
}

void type_check_integer_args(const PrimExpr& lhs, const PrimExpr& rhs, const char* op) {
  ICHECK(lhs.dtype().is_int() || lhs.dtype().is_uint())
      << "Expected integer argument as LHS of " << op << ", but received " << lhs << " of type "
      << lhs.dtype();
  ICHECK(rhs.dtype().is_int() || rhs.dtype().is_uint())
      << "Expected integer argument as RHS of " << op << ", but received " << rhs << " of type "
      << rhs.dtype();
}
}  // namespace

PrimExpr operator&&(PrimExpr a, PrimExpr b) { return logical_and(a, b); }
PrimExpr logical_and(PrimExpr a, PrimExpr b, Span span) {
  type_check_boolean_args(a, b, "&& operator (logical AND)");
  if (auto ret = arith::TryConstFold<tir::And>(a, b)) return ret.value();
  return tir::And(a, b, span);
}

PrimExpr operator||(PrimExpr a, PrimExpr b) { return logical_or(a, b); }
PrimExpr logical_or(PrimExpr a, PrimExpr b, Span span) {
  type_check_boolean_args(a, b, "|| operator (logical OR)");
  if (auto ret = arith::TryConstFold<tir::Or>(a, b)) return ret.value();
  return tir::Or(a, b, span);
}

PrimExpr operator!(PrimExpr a) { return logical_not(a); }
PrimExpr logical_not(PrimExpr a, Span span) {
  type_check_boolean_args(a, "! operator (logical NOT)");
  if (auto ret = arith::TryConstFold<tir::Not>(a)) return ret.value();
  return tir::Not(a, span);
}

// shift right
PrimExpr operator>>(PrimExpr a, PrimExpr b) { return right_shift(a, b); }

PrimExpr right_shift(PrimExpr a, PrimExpr b, Span span) {
  type_check_integer_args(a, b, ">> operator (right shift)");

  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pb)
      ICHECK(pb->value >= 0 && pb->value < rtype.bits())
          << "Shift amount must be non-negative and less than " << rtype.bits() << " for type "
          << rtype;
    if (pa && pb) {
      return IntImm(rtype, (pa->value >> pb->value), span);
    }
    if (pb) {
      if (pb->value == 0) return a;
    }
  });

  return tir::Call(a.dtype(), tir::builtin::shift_right(), {a, b}, span);
}

// shift left
PrimExpr operator<<(PrimExpr a, PrimExpr b) { return left_shift(a, b); }
PrimExpr left_shift(PrimExpr a, PrimExpr b, Span span) {
  type_check_integer_args(a, b, "<< operator (left shift)");
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pb)
      ICHECK(pb->value >= 0 && pb->value < rtype.bits())
          << "Shift amount must be non-negative and less than " << rtype.bits() << " for type "
          << rtype;
    if (pa && pb) return IntImm(rtype, (pa->value << pb->value), span);
    if (pb) {
      if (pb->value == 0) return a;
    }
  });
  return tir::Call(a.dtype(), tir::builtin::shift_left(), {a, b}, span);
}

// bitwise and
PrimExpr operator&(PrimExpr a, PrimExpr b) { return bitwise_and(a, b); }
PrimExpr bitwise_and(PrimExpr a, PrimExpr b, Span span) {
  type_check_integer_args(a, b, "& operator (bitwise AND)");
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, (pa->value & pb->value), span);
  });
  return tir::Call(a.dtype(), tir::builtin::bitwise_and(), {a, b}, span);
}

// bitwise_or
PrimExpr operator|(PrimExpr a, PrimExpr b) { return bitwise_or(a, b); }
PrimExpr bitwise_or(PrimExpr a, PrimExpr b, Span span) {
  type_check_integer_args(a, b, "| operator (bitwise OR)");
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, (pa->value | pb->value), span);
  });
  return tir::Call(a.dtype(), tir::builtin::bitwise_or(), {a, b}, span);
}

// bitwise_xor
PrimExpr operator^(PrimExpr a, PrimExpr b) { return bitwise_xor(a, b); }
PrimExpr bitwise_xor(PrimExpr a, PrimExpr b, Span span) {
  type_check_integer_args(a, b, "^ operator (bitwise XOR)");
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, (pa->value ^ pb->value), span);
  });
  return tir::Call(a.dtype(), tir::builtin::bitwise_xor(), {a, b}, span);
}

// bitwise_not
PrimExpr operator~(PrimExpr a) { return bitwise_neg(a); }

PrimExpr bitwise_neg(PrimExpr a, Span span) {
  type_check_integer_args(a, "~ operator (bitwise NOT)");
  return tir::Call(a.dtype(), tir::builtin::bitwise_not(), {a}, span);
}

TVM_REGISTER_GLOBAL("tir.bitwise_not").set_body_typed([](PrimExpr a, Span span) {
  return bitwise_neg(a, span);
});

// pow
PrimExpr pow(PrimExpr x, PrimExpr y, Span span) {
  BinaryOpMatchTypes(x, y, span);
  ICHECK(x.dtype().is_float()) << "power only applies to float";

  // If we detect pow(x, 3), suggest using x * x * x
  if (y.dtype().is_int()) {
    using tir::IntImmNode;
    const IntImmNode* px = y.as<IntImmNode>();
    if (px) {
      if (px->value >= 3) {
        LOG(WARNING)
            << "Detected pow(x, y) where y >= 3, it is recommended to avoid this as it may lead to "
               "uninteded behaviors when x < 0. Perhaps with `x * x * x ...` or "
               "`pow(x, 2) * pow(x, 2) ...`.";
      }
    }
  } else if (y.dtype().is_float()) {
    using tir::FloatImmNode;
    const FloatImmNode* fx = y.as<FloatImmNode>();
    if (fx) {
      if (fx->value >= 3.0) {
        LOG(WARNING)
            << "Detected pow(x, y) where y >= 3, it is recommended to avoid this as it may lead to "
               "uninteded behaviors when x < 0. Perhaps with `x * x * x ...` or "
               "`pow(x, 2) * pow(x, 2) ...`.";
      }
    }
  }

  static auto op = Op::Get("tir.pow");
  return tir::Call(x.dtype(), op, {x, y}, span);
}

TVM_TIR_REGISTER_PURE_BINARY_OP("pow").set_attr<TVectorizable>("TVectorizable", true);

// abs
PrimExpr abs(PrimExpr x, Span span) {
  if (x.dtype().is_int()) {
    using tir::IntImmNode;
    const IntImmNode* px = x.as<IntImmNode>();
    if (px) {
      return IntImm(x.dtype(), std::abs(px->value), px->span);
    }
    return tir::Select(x >= make_zero(x.dtype()), x, -x, span);
  } else if (x.dtype().is_float()) {
    using tir::FloatImmNode;
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return FloatImm(x.dtype(), std::fabs(fx->value), fx->span);
    }
    static auto op = Op::Get("tir.fabs");
    return tir::Call(x.dtype(), op, {x}, span);
  } else if (x.dtype().is_uint()) {
    return x;
  } else {
    LOG(FATAL) << "Data type " << x.dtype()
               << " not supported for absolute op. Skipping absolute op...";
    return x;
  }
}

TVM_TIR_REGISTER_PURE_UNARY_OP("fabs").set_attr<TVectorizable>("TVectorizable", true);

// isnan
PrimExpr isnan(PrimExpr x, Span span) {
  DataType t = DataType::Bool(x.dtype().lanes());
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return make_const(t, false);
  } else if (x.dtype().is_float()) {
    using tir::FloatImmNode;
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return make_const(t, std::isnan(fx->value), fx->span);
    }
    static auto op = Op::Get("tir.isnan");
    if (x.dtype().bits() == 16) {
      return tir::Call(t, op, {cast(DataType::Float(32, t.lanes()), std::move(x), span)}, span);
    } else {
      return tir::Call(t, op, {x}, span);
    }
  } else {
    LOG(FATAL) << "Data type " << x.dtype() << " not supported for isnan op. Skipping isnan op...";
  }
}

// isinf
PrimExpr isinf(PrimExpr x, Span span) {
  DataType t = DataType::Bool(x.dtype().lanes());
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return make_const(t, false, span);
  } else if (x.dtype().is_float()) {
    PrimExpr infX = infinity(x.dtype(), span);
    return abs(x, span) == infX && !isnan(x, span);
  } else {
    LOG(FATAL) << "Data type " << x.dtype() << " not supported for finiteness ops. Skipping it...";
  }
}

// isfinite
PrimExpr isfinite(PrimExpr x, Span span) { return !isinf(x, span) && !isnan(x, span); }

PrimExpr sum(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  Var x("x", source.dtype(), span), y("y", source.dtype(), span);
  PrimExpr result = tir::Add(x, y, span);
  PrimExpr identity_element = make_zero(source.dtype(), span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

PrimExpr all(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  type_check_boolean_args(source, "tvm::all");
  Var x("x", source.dtype(), span), y("y", source.dtype());
  PrimExpr result = tir::And(x, y, span);
  PrimExpr identity_element = make_const(source.dtype(), true, span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

PrimExpr any(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  type_check_boolean_args(source, "tvm::any");
  Var x("x", source.dtype(), span), y("y", source.dtype(), span);
  PrimExpr result = tir::Or(x, y, span);
  PrimExpr identity_element = make_const(source.dtype(), false, span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

PrimExpr max(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  Var x("x", source.dtype(), span), y("y", source.dtype(), span);
  PrimExpr result = tir::Max(x, y, span);
  PrimExpr identity_element = min_value(source.dtype(), span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

PrimExpr min(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  Var x("x", source.dtype(), span), y("y", source.dtype(), span);
  PrimExpr result = tir::Min(x, y, span);
  PrimExpr identity_element = max_value(source.dtype(), span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

PrimExpr prod(PrimExpr source, Array<IterVar> rdom, Array<PrimExpr> init, Span span) {
  Var x("x", source.dtype(), span), y("y", source.dtype(), span);
  PrimExpr result = tir::Mul(x, y, span);
  PrimExpr identity_element = make_const(source.dtype(), 1, span);
  tir::CommReducer combiner = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tir::Reduce(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0, init, span);
}

// fmod
PrimExpr fmod(PrimExpr x, PrimExpr y, Span span) {
  BinaryOpMatchTypes(x, y, span);
  ICHECK(x.dtype().is_float()) << "fmod only applies to float";
  static auto op = Op::Get("tir.fmod");
  return tir::Call(x.dtype(), op, {x, y}, span);
}

TVM_TIR_REGISTER_PURE_UNARY_OP("fmod");

// floor
PrimExpr floor(PrimExpr x, Span span) {
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return x;
  }
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::floor(fx->value), fx->span);
  static auto op = Op::Get("tir.floor");
  return tir::Call(x.dtype(), op, {x}, span);
}

TVM_TIR_REGISTER_PURE_UNARY_OP("floor").set_attr<TVectorizable>("TVectorizable", true);

// ceil
PrimExpr ceil(PrimExpr x, Span span) {
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return x;
  }
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::ceil(fx->value), fx->span);
  static auto op = Op::Get("tir.ceil");
  return tir::Call(x.dtype(), op, {x}, span);
}

TVM_TIR_REGISTER_PURE_UNARY_OP("ceil").set_attr<TVectorizable>("TVectorizable", true);

// round
PrimExpr round(PrimExpr x, Span span) {
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return x;
  }
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::nearbyint(fx->value), fx->span);
  static auto op = Op::Get("tir.round");
  return tir::Call(x.dtype(), op, {x}, span);
}

TVM_TIR_REGISTER_PURE_UNARY_OP("round").set_attr<TVectorizable>("TVectorizable", true);

// nearbyint
PrimExpr nearbyint(PrimExpr x, Span span) {
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return x;
  }
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::nearbyint(fx->value), fx->span);
  static auto op = Op::Get("tir.nearbyint");
  return tir::Call(x.dtype(), op, {x}, span);
}

TVM_TIR_REGISTER_PURE_UNARY_OP("nearbyint");

// trunc
PrimExpr trunc(PrimExpr x, Span span) {
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return x;
  }
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) {
    return FloatImm(x.dtype(), (fx->value < 0 ? std::ceil(fx->value) : std::floor(fx->value)),
                    fx->span);
  }
  static auto op = Op::Get("tir.trunc");
  return tir::Call(x.dtype(), op, {x}, span);
}

TVM_TIR_REGISTER_PURE_UNARY_OP("trunc").set_attr<TVectorizable>("TVectorizable", true);

// unary op registration.
TVM_TIR_REGISTER_PURE_UNARY_OP("exp").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("exp2").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("exp10").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("erf");

TVM_TIR_REGISTER_PURE_UNARY_OP("tanh").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("sigmoid").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("sqrt").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("rsqrt");

TVM_TIR_REGISTER_PURE_UNARY_OP("log").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("log2").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("log1p");

TVM_TIR_REGISTER_PURE_UNARY_OP("log10").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("tan").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("cos").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("cosh").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("sin").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("sinh").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("asin");

TVM_TIR_REGISTER_PURE_UNARY_OP("acos");

TVM_TIR_REGISTER_PURE_UNARY_OP("atan");

TVM_TIR_REGISTER_PURE_UNARY_OP("acosh");

TVM_TIR_REGISTER_PURE_UNARY_OP("asinh");

TVM_TIR_REGISTER_PURE_UNARY_OP("atanh");

TVM_TIR_REGISTER_PURE_UNARY_OP("clz");

// binary intrinsics
TVM_TIR_REGISTER_PURE_BINARY_OP("atan2");

TVM_TIR_REGISTER_PURE_BINARY_OP("nextafter");

TVM_TIR_REGISTER_PURE_BINARY_OP("hypot");

TVM_TIR_REGISTER_PURE_BINARY_OP("copysign");

TVM_TIR_REGISTER_PURE_BINARY_OP("ldexp");

TVM_TIR_REGISTER_OP("TVMBackendAllocWorkspace")
    .set_num_inputs(5)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "TVMBackendAllocWorkspace")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_TIR_REGISTER_OP("TVMBackendFreeWorkspace")
    .set_num_inputs(3)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "TVMBackendFreeWorkspace")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

// expose basic functions to node namespace
TVM_REGISTER_GLOBAL("node._const").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (auto opt = args[0].TryAsInt()) {
    *ret = tir::make_const(args[1], opt.value(), args[2]);
  } else if (auto opt = args[0].TryAsBool()) {
    *ret = tir::make_const(args[1], opt.value(), args[2]);
  } else if (auto opt = args[0].TryAsFloat()) {
    *ret = tir::make_const(args[1], opt.value(), args[2]);
  } else {
    LOG(FATAL) << "First argument to tvm.tir.const must be int, float, or bool, "
               << "but instead received argument with type code " << args[0].type_code();  // FIXME
  }
});

TVM_REGISTER_GLOBAL("node.LargeUIntImm").set_body_typed(LargeUIntImm);

TVM_REGISTER_GLOBAL("tir.min_value").set_body_typed(min_value);

TVM_REGISTER_GLOBAL("tir.max_value").set_body_typed(max_value);

TVM_REGISTER_GLOBAL("tir.infinity").set_body_typed(infinity);

TVM_REGISTER_GLOBAL("tir.abs").set_body_typed(tvm::abs);

TVM_REGISTER_GLOBAL("tir.likely").set_body_typed(tvm::likely);

TVM_REGISTER_GLOBAL("tir.isnan").set_body_typed(tvm::isnan);

TVM_REGISTER_GLOBAL("tir.isfinite").set_body_typed(tvm::isfinite);

TVM_REGISTER_GLOBAL("tir.isinf").set_body_typed(tvm::isinf);

TVM_REGISTER_GLOBAL("tir.floor").set_body_typed(tvm::floor);

TVM_REGISTER_GLOBAL("tir.ceil").set_body_typed(tvm::ceil);

TVM_REGISTER_GLOBAL("tir.round").set_body_typed(tvm::round);

TVM_REGISTER_GLOBAL("tir.nearbyint").set_body_typed(tvm::nearbyint);

TVM_REGISTER_GLOBAL("tir.trunc").set_body_typed(tvm::trunc);

TVM_REGISTER_GLOBAL("tir._cast").set_body_typed(tvm::cast);

TVM_REGISTER_GLOBAL("tir.reinterpret").set_body_typed(tvm::reinterpret);

// operator overloading, smarter than make
#define REGISTER_MAKE_BINARY_OP(Node, Func)                                                \
  TVM_REGISTER_GLOBAL("tir." #Node).set_body_typed([](PrimExpr a, PrimExpr b, Span span) { \
    return (Func(a, b, span));                                                             \
  })

#define REGISTER_MAKE_BIT_OP(Node, Func)                                                \
  TVM_REGISTER_GLOBAL("tir." #Node).set_body([](TVMArgs args, TVMRetValue* ret) {       \
    bool lhs_is_int = args[0].type_code() == kDLInt;                                    \
    bool rhs_is_int = args[1].type_code() == kDLInt;                                    \
    if (lhs_is_int) {                                                                   \
      *ret = (Func(args[0].operator int(), args[1].operator PrimExpr(), args[2]));      \
    } else if (rhs_is_int) {                                                            \
      *ret = (Func(args[0].operator PrimExpr(), args[1].operator int(), args[2]));      \
    } else {                                                                            \
      *ret = (Func(args[0].operator PrimExpr(), args[1].operator PrimExpr(), args[2])); \
    }                                                                                   \
  })

REGISTER_MAKE_BINARY_OP(_OpAdd, add);
REGISTER_MAKE_BINARY_OP(_OpSub, sub);
REGISTER_MAKE_BINARY_OP(_OpMul, mul);
REGISTER_MAKE_BINARY_OP(_OpDiv, div);
REGISTER_MAKE_BINARY_OP(_OpMod, truncmod);
REGISTER_MAKE_BINARY_OP(_OpIndexDiv, indexdiv);
REGISTER_MAKE_BINARY_OP(_OpIndexMod, indexmod);
REGISTER_MAKE_BINARY_OP(_OpFloorDiv, floordiv);
REGISTER_MAKE_BINARY_OP(_OpFloorMod, floormod);
REGISTER_MAKE_BINARY_OP(_OpTruncDiv, truncdiv);
REGISTER_MAKE_BINARY_OP(_OpTruncMod, truncmod);
REGISTER_MAKE_BINARY_OP(_OpCeilDiv, ceildiv);
REGISTER_MAKE_BINARY_OP(_OpPow, pow);
REGISTER_MAKE_BINARY_OP(_OpMin, min);
REGISTER_MAKE_BINARY_OP(_OpMax, max);
REGISTER_MAKE_BINARY_OP(_OpEQ, equal);
REGISTER_MAKE_BINARY_OP(_OpNE, not_equal);
REGISTER_MAKE_BINARY_OP(_OpLT, less);        // NOLINT(*)
REGISTER_MAKE_BINARY_OP(_OpLE, less_equal);  // NOLINT(*)
REGISTER_MAKE_BINARY_OP(_OpGT, greater);     // NOLINT(*)
REGISTER_MAKE_BINARY_OP(_OpGE, greater_equal);
REGISTER_MAKE_BINARY_OP(_OpAnd, logical_and);
REGISTER_MAKE_BINARY_OP(_OpOr, logical_or);
REGISTER_MAKE_BIT_OP(bitwise_and, bitwise_and);
REGISTER_MAKE_BIT_OP(bitwise_or, bitwise_or);
REGISTER_MAKE_BIT_OP(bitwise_xor, bitwise_xor);
REGISTER_MAKE_BIT_OP(left_shift, left_shift);  // NOLINT(*)
REGISTER_MAKE_BIT_OP(right_shift, right_shift);

TVM_REGISTER_GLOBAL("tir._OpIfThenElse")
    .set_body_typed([](PrimExpr cond, PrimExpr true_value, PrimExpr false_value, Span span) {
      return if_then_else(cond, true_value, false_value, span);
    });

TVM_REGISTER_GLOBAL("tir.const_true").set_body_typed([](DataType t, Span span) {
  return const_true(t.lanes(), span);
});

PrimExpr fast_erf_float_expr(PrimExpr arg, int bits) {
  auto plus_4 = make_const(DataType::Float(bits), 4.f);
  auto minus_4 = make_const(DataType::Float(bits), -4.f);

  // The monomial coefficients of the numerator polynomial (odd).
  auto alpha_1 = make_const(DataType::Float(bits), -1.60960333262415e-02f);
  auto alpha_3 = make_const(DataType::Float(bits), -2.95459980854025e-03f);
  auto alpha_5 = make_const(DataType::Float(bits), -7.34990630326855e-04f);
  auto alpha_7 = make_const(DataType::Float(bits), -5.69250639462346e-05f);
  auto alpha_9 = make_const(DataType::Float(bits), -2.10102402082508e-06f);
  auto alpha_11 = make_const(DataType::Float(bits), 2.77068142495902e-08f);
  auto alpha_13 = make_const(DataType::Float(bits), -2.72614225801306e-10f);

  // The monomial coefficients of the denominator polynomial (even).
  auto beta_0 = make_const(DataType::Float(bits), -1.42647390514189e-02f);
  auto beta_2 = make_const(DataType::Float(bits), -7.37332916720468e-03f);
  auto beta_4 = make_const(DataType::Float(bits), -1.68282697438203e-03f);
  auto beta_6 = make_const(DataType::Float(bits), -2.13374055278905e-04f);
  auto beta_8 = make_const(DataType::Float(bits), -1.45660718464996e-05f);

  // clamp x
  auto x = tvm::max(tvm::min(arg, plus_4), minus_4);
  auto x2 = x * x;

  // Evaluate the numerator polynomial p.
  auto p = x2 * alpha_13 + alpha_11;
  p = x2 * p + alpha_9;
  p = x2 * p + alpha_7;
  p = x2 * p + alpha_5;
  p = x2 * p + alpha_3;
  p = x2 * p + alpha_1;
  p = x * p;

  // Evaluate the denominator polynomial p.
  auto q = x2 * beta_8 + beta_6;
  q = x2 * q + beta_4;
  q = x2 * q + beta_2;
  q = x2 * q + beta_0;

  return p / q;
}

}  // namespace tvm
