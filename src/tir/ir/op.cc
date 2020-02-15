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
 * \file expr_operator.cc
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <cmath>
// Centralized header for constant folders.
#include "../../arith/const_fold.h"

namespace tvm {

using namespace tir;

// simple cast that only checks if type matches and cast
inline PrimExpr SimpleCast(const DataType& t, PrimExpr value) {
  if (value.dtype() == t) return value;
  return tir::CastNode::make(t, value);
}

PrimExpr LargeUIntImm(DataType t, int64_t low, int64_t high) {
  return tir::CallNode::make(
      t, tir::intrinsic::tvm_large_uint_imm,
      {make_const(DataType::UInt(32), low),
       make_const(DataType::UInt(32), high)},
      tir::CallNode::PureIntrinsic);
}

// The public function with a quick checking path.
void BinaryOpMatchTypes(PrimExpr& lhs, PrimExpr& rhs) {  // NOLINT(*)
  if (lhs.dtype() == rhs.dtype()) return;
  DataType ltype = lhs.dtype();
  DataType rtype = rhs.dtype();
  if (ltype.lanes() == 1 && rtype.lanes() != 1) {
    lhs = tir::BroadcastNode::make(lhs, rtype.lanes());
  } else if (rtype.lanes() == 1 && ltype.lanes() != 1) {
    rhs = tir::BroadcastNode::make(rhs, ltype.lanes());
  } else {
    CHECK(ltype.lanes() == rtype.lanes())
        << "Cannot match type " << ltype << " vs " << rtype;
  }
  if (lhs.dtype() == rhs.dtype()) return;
  // Only do very simple type coversion
  // int->float, DataType::Int(32)->int(64)
  // require the types to be relatively consistent
  // This will the reduce amount code generated by operators
  // and also help user to find potential type conversion problems.
  if (!lhs.dtype().is_float() && rhs.dtype().is_float()) {
    // int->float
    lhs = cast(rhs.dtype(), lhs);
  } else if (lhs.dtype().is_float() && !rhs.dtype().is_float()) {
    // int->float
    rhs = cast(lhs.dtype(), rhs);
  } else if ((lhs.dtype().is_int() && rhs.dtype().is_int()) ||
             (lhs.dtype().is_uint() && rhs.dtype().is_uint())) {
    // promote int to higher bits
    if (lhs.dtype().bits() < rhs.dtype().bits()) {
      lhs = cast(rhs.dtype(), lhs);
    } else {
      rhs = cast(lhs.dtype(), rhs);
    }
  } else if ((lhs.dtype().is_int() && rhs.dtype().is_uint()) ||
             (lhs.dtype().is_uint() && rhs.dtype().is_int())) {
    int bits = std::max(lhs.dtype().bits(), rhs.dtype().bits());
    lhs = SimpleCast(DataType::Int(bits, lhs.dtype().lanes()), lhs);
    rhs = SimpleCast(DataType::Int(bits, rhs.dtype().lanes()), rhs);
  } else {
    LOG(FATAL) << "Cannot match type " << ltype << " vs " << rtype;
  }
}

// maximum and min limits
PrimExpr max_value(const DataType& dtype) {
  using namespace tir;
  CHECK_EQ(dtype.lanes(), 1);
  if (dtype.is_int()) {
    if (dtype.bits() == 64) {
      return IntImm(dtype, std::numeric_limits<int64_t>::max());
    } else if (dtype.bits() < 64) {
      int64_t val = 1;
      val = (val << (dtype.bits() - 1)) - 1;
      return IntImm(dtype, val);
    }
  } else if (dtype.is_uint()) {
    if (dtype.bits() == 64) {
      return make_const(dtype, std::numeric_limits<uint64_t>::max());
    } else if (dtype.bits() < 64) {
      uint64_t val = 1;
      val = (val << static_cast<uint64_t>(dtype.bits())) - 1;
      return IntImm(dtype, static_cast<int64_t>(val));
    }
  } else if (dtype.is_float()) {
    if (dtype.bits() == 64) {
      return FloatImm(dtype, std::numeric_limits<double>::max());
    } else if (dtype.bits() == 32) {
      return FloatImm(dtype, std::numeric_limits<float>::max());
    } else if (dtype.bits() == 16) {
      return FloatImm(dtype, 65504.0);
    }
  }
  LOG(FATAL) << "Cannot decide max_value for type" << dtype;
  return PrimExpr();
}

PrimExpr min_value(const DataType& dtype) {
  using namespace tir;
  CHECK_EQ(dtype.lanes(), 1);
  if (dtype.is_int()) {
    if (dtype.bits() == 64) {
      return IntImm(dtype, std::numeric_limits<int64_t>::lowest());
    } else if (dtype.bits() < 64) {
      int64_t val = 1;
      val = -(val << (dtype.bits() - 1));
      return IntImm(dtype, val);
    }
  } else if (dtype.is_uint()) {
    return IntImm(dtype, 0);
  } else if (dtype.is_float()) {
    if (dtype.bits() == 64) {
      return FloatImm(dtype, std::numeric_limits<double>::lowest());
    } else if (dtype.bits() == 32) {
      return FloatImm(dtype, std::numeric_limits<float>::lowest());
    } else if (dtype.bits() == 16) {
      return FloatImm(dtype, -65504.0);
    }
  }
  LOG(FATAL) << "Cannot decide min_value for type" << dtype;
  return PrimExpr();
}

namespace tir {
template<typename ValueType>
inline bool ConstPowerHelper(ValueType val, int *shift) {
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

PrimExpr cast(const DataType& t, PrimExpr value) {
  using tir::FloatImmNode;
  if (value.dtype() == t) return value;
  // const fold IntImm as they are used in index computations
  if (t.lanes() == 1) {
    if (const IntImmNode* op = value.as<IntImmNode>()) {
      return make_const(t, op->value);
    } else if (const FloatImmNode* op = value.as<FloatImmNode>()) {
      return make_const(t, op->value);
    }
    return tir::CastNode::make(t, value);
  } else {
    if (value.dtype().lanes() == 1) {
      // manually unroll cast
      DataType vtype = t.element_of();
      if (value.dtype() != vtype) {
        if (const IntImmNode* op = value.as<IntImmNode>()) {
          value = make_const(vtype, op->value);
        } else if (const FloatImmNode* op = value.as<FloatImmNode>()) {
          value = make_const(vtype, op->value);
        } else {
          value = tir::CastNode::make(vtype, value);
        }
      }
      return tir::BroadcastNode::make(value, t.lanes());
    } else {
      CHECK(value.dtype().lanes() == t.lanes());
      return tir::CastNode::make(t, value);
    }
  }
}

PrimExpr reinterpret(const DataType& t, PrimExpr value) {
  if (value.dtype() == t) return value;
  return tir::CallNode::make(
    t, tir::CallNode::reinterpret, { value }, tir::CallNode::PureIntrinsic);
}

PrimExpr operator+(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::AddNode>(a, b);
  if (ret.defined()) return ret;
  return tir::AddNode::make(a, b);
}

// negation
PrimExpr operator-(PrimExpr a) {
  using tir::IntImmNode;
  using tir::FloatImmNode;
  const IntImmNode* pa = a.as<IntImmNode>();
  const FloatImmNode* fa = a.as<FloatImmNode>();
  if (pa) return IntImm(a.dtype(), -pa->value);
  if (fa) return FloatImm(a.dtype(), -fa->value);
  return make_zero(a.dtype()) - a;
}

PrimExpr operator-(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::SubNode>(a, b);
  if (ret.defined()) return ret;
  return tir::SubNode::make(a, b);
}

PrimExpr operator*(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::MulNode>(a, b);
  if (ret.defined()) return ret;
  return tir::MulNode::make(a, b);
}

PrimExpr div(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::DivNode>(a, b);
  if (ret.defined()) return ret;
  return tir::DivNode::make(a, b);
}

PrimExpr truncdiv(PrimExpr a, PrimExpr b) {
  CHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  CHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  return div(a, b);
}

PrimExpr truncmod(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::ModNode>(a, b);
  if (ret.defined()) return ret;
  return tir::ModNode::make(a, b);
}

PrimExpr operator/(PrimExpr a, PrimExpr b) {
  return div(a, b);
}

PrimExpr operator%(PrimExpr a, PrimExpr b) {
  return truncmod(a, b);
}

// TODO(tqchen): switch to floordiv
PrimExpr indexdiv(PrimExpr a, PrimExpr b) {
  return floordiv(a, b);
}

PrimExpr indexmod(PrimExpr a, PrimExpr b) {
  return floormod(a, b);
}

PrimExpr floordiv(PrimExpr a, PrimExpr b) {
  CHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  CHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::FloorDivNode>(a, b);
  if (ret.defined()) return ret;
  return tir::FloorDivNode::make(a, b);
}

PrimExpr floormod(PrimExpr a, PrimExpr b) {
  CHECK(a.dtype().is_int() || a.dtype().is_uint()) << a;
  CHECK(b.dtype().is_int() || b.dtype().is_uint()) << b;
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::FloorModNode>(a, b);
  if (ret.defined()) return ret;
  return tir::FloorModNode::make(a, b);
}

PrimExpr min(PrimExpr a, PrimExpr b) {
  // inf-aware simplificaiton
  using arith::is_pos_inf;
  using arith::is_neg_inf;
  if (is_pos_inf(a)) return b;
  if (is_neg_inf(a)) return a;
  if (is_pos_inf(b)) return a;
  if (is_neg_inf(b)) return b;
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::MinNode>(a, b);
  if (ret.defined()) return ret;
  return tir::MinNode::make(a, b);
}

PrimExpr max(PrimExpr a, PrimExpr b) {
  // inf-aware simplificaiton
  using arith::is_pos_inf;
  using arith::is_neg_inf;
  if (is_pos_inf(a)) return a;
  if (is_neg_inf(a)) return b;
  if (is_pos_inf(b)) return b;
  if (is_neg_inf(b)) return a;
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::MaxNode>(a, b);
  if (ret.defined()) return ret;
  return tir::MaxNode::make(a, b);
}

PrimExpr if_then_else(PrimExpr cond, PrimExpr true_value, PrimExpr false_value) {
  CHECK(cond.dtype() == DataType::Bool(1))
      << "if_then_else only accept the condition to be boolean type.";
  BinaryOpMatchTypes(true_value, false_value);
  if (const IntImmNode* op = cond.as<IntImmNode>()) {
    if (op->value != 0) {
      return true_value;
    } else {
      return false_value;
    }
  }
  return tir::CallNode::make(
      true_value.dtype(),
      tir::intrinsic::tvm_if_then_else,
      {cond, true_value, false_value},
      tir::CallNode::PureIntrinsic);
}

PrimExpr likely(PrimExpr cond) {
  if (is_const(cond)) return cond;
  return tir::CallNode::make(cond.dtype(),
                            tir::CallNode::likely,
                            { cond },
                            tir::CallNode::PureIntrinsic);
}

PrimExpr operator>(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::GTNode>(a, b);
  if (ret.defined()) return ret;
  return tir::GTNode::make(a, b);
}

PrimExpr operator>=(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::GENode>(a, b);
  if (ret.defined()) return ret;
  return tir::GENode::make(a, b);
}

PrimExpr operator<(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::LTNode>(a, b);
  if (ret.defined()) return ret;
  return tir::LTNode::make(a, b);
}

PrimExpr operator<=(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::LENode>(a, b);
  if (ret.defined()) return ret;
  return tir::LENode::make(a, b);
}

PrimExpr operator==(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::EQNode>(a, b);
  if (ret.defined()) return ret;
  return tir::EQNode::make(a, b);
}

PrimExpr operator!=(PrimExpr a, PrimExpr b) {
  BinaryOpMatchTypes(a, b);
  PrimExpr ret = arith::TryConstFold<tir::NENode>(a, b);
  if (ret.defined()) return ret;
  return tir::NENode::make(a, b);
}

PrimExpr operator&&(PrimExpr a, PrimExpr b) {
  CHECK(a.dtype().is_bool());
  CHECK(b.dtype().is_bool());
  PrimExpr ret = arith::TryConstFold<tir::AndNode>(a, b);
  if (ret.defined()) return ret;
  return tir::AndNode::make(a, b);
}

PrimExpr operator||(PrimExpr a, PrimExpr b) {
  CHECK(a.dtype().is_bool());
  CHECK(b.dtype().is_bool());
  PrimExpr ret = arith::TryConstFold<tir::OrNode>(a, b);
  if (ret.defined()) return ret;
  return tir::OrNode::make(a, b);
}

PrimExpr operator!(PrimExpr a) {
  CHECK(a.dtype().is_bool());
  PrimExpr ret = arith::TryConstFold<tir::NotNode>(a);
  if (ret.defined()) return ret;
  return tir::NotNode::make(a);
}

PrimExpr operator>>(PrimExpr a, PrimExpr b) {
  CHECK(a.dtype().is_int() || a.dtype().is_uint());
  CHECK(b.dtype().is_int() || b.dtype().is_uint());
  BinaryOpMatchTypes(a, b);
  TVM_INDEX_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) return IntImm(rtype, (pa->value >> pb->value));
      if (pb) {
        if (pb->value == 0) return a;
      }
    });
  return tir::CallNode::make(
    a.dtype(), tir::CallNode::shift_right, { a, b }, tir::CallNode::PureIntrinsic);
}

PrimExpr operator<<(PrimExpr a, PrimExpr b) {
  CHECK(a.dtype().is_int() || a.dtype().is_uint());
  CHECK(b.dtype().is_int() || b.dtype().is_uint());
  BinaryOpMatchTypes(a, b);
  TVM_INDEX_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) return IntImm(rtype, (pa->value << pb->value));
      if (pb) {
        if (pb->value == 0) return a;
      }
    });
  return tir::CallNode::make(
    a.dtype(), tir::CallNode::shift_left, { a, b }, tir::CallNode::PureIntrinsic);
}

PrimExpr operator&(PrimExpr a, PrimExpr b) {
  CHECK(a.dtype().is_int() || a.dtype().is_uint());
  CHECK(b.dtype().is_int() || b.dtype().is_uint());
  BinaryOpMatchTypes(a, b);
  TVM_INDEX_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) return IntImm(rtype, (pa->value & pb->value));
    });
  return tir::CallNode::make(
    a.dtype(), tir::CallNode::bitwise_and, { a, b }, tir::CallNode::PureIntrinsic);
}

PrimExpr operator|(PrimExpr a, PrimExpr b) {
  CHECK(a.dtype().is_int() || a.dtype().is_uint());
  CHECK(b.dtype().is_int() || b.dtype().is_uint());
  BinaryOpMatchTypes(a, b);
  TVM_INDEX_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) return IntImm(rtype, (pa->value | pb->value));
    });
  return tir::CallNode::make(
    a.dtype(), tir::CallNode::bitwise_or, { a, b }, tir::CallNode::PureIntrinsic);
}

PrimExpr operator^(PrimExpr a, PrimExpr b) {
  CHECK(a.dtype().is_int() || a.dtype().is_uint());
  CHECK(b.dtype().is_int() || b.dtype().is_uint());
  BinaryOpMatchTypes(a, b);
  TVM_INDEX_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) return IntImm(rtype, (pa->value ^ pb->value));
    });
  return tir::CallNode::make(
    a.dtype(), tir::CallNode::bitwise_xor, { a, b }, tir::CallNode::PureIntrinsic);
}

PrimExpr operator~(PrimExpr a) {
  CHECK(a.dtype().is_int() || a.dtype().is_uint());
  return tir::CallNode::make(
    a.dtype(), tir::CallNode::bitwise_not, { a }, tir::CallNode::PureIntrinsic);
}

PrimExpr pow(PrimExpr x, PrimExpr y) {
  BinaryOpMatchTypes(x, y);
  CHECK(x.dtype().is_float()) << "power only applies to float";
  return tir::CallNode::make(
    x.dtype(), "pow", { x, y }, tir::CallNode::PureIntrinsic);
}

PrimExpr abs(PrimExpr x) {
  if (x.dtype().is_int()) {
    using tir::IntImmNode;
    const IntImmNode* px = x.as<IntImmNode>();
    if (px) {
      return IntImm(x.dtype(), std::abs(px->value));
    }
    return tir::SelectNode::make(x >= make_zero(x.dtype()), x, -x);
  } else if (x.dtype().is_float()) {
    using tir::FloatImmNode;
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return FloatImm(x.dtype(), std::fabs(fx->value));
    }
    return tir::CallNode::make(x.dtype(), "fabs", {x}, tir::CallNode::PureIntrinsic);
  } else if (x.dtype().is_uint()) {
    return x;
  } else {
    LOG(FATAL) << "Data type " << x.dtype()
               <<" not supported for absolute op. Skipping absolute op...";
    return x;
  }
}

PrimExpr isnan(PrimExpr x) {
  DataType t = DataType::Bool(x.dtype().lanes());
  if (x.dtype().is_int() || x.dtype().is_uint()) {
    return make_const(t, false);
  } else if (x.dtype().is_float()) {
    using tir::FloatImmNode;
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return make_const(t, std::isnan(fx->value));
    }
    if (x.dtype().bits() == 16) {
      return tir::CallNode::make(t, tir::CallNode::isnan,
                               {cast(DataType::Float(32, t.lanes()), std::move(x))},
                               tir::CallNode::PureIntrinsic);
    } else {
      return tir::CallNode::make(t, tir::CallNode::isnan, {x}, tir::CallNode::PureIntrinsic);
    }
  } else {
    LOG(FATAL) << "Data type " << x.dtype()
               <<" not supported for isnan op. Skipping isnan op...";
    return x;
  }
}

PrimExpr sum(PrimExpr source, Array<IterVar> rdom) {
  Var x("x", source.dtype()), y("y", source.dtype());
  PrimExpr result = tir::AddNode::make(x, y);
  PrimExpr identity_element = make_zero(source.dtype());
  tir::CommReducer combiner =
    tir::CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return tir::ReduceNode::make(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0);
}

PrimExpr all(PrimExpr source, Array<IterVar> rdom) {
  CHECK(source.dtype().is_bool());
  Var x("x", source.dtype()), y("y", source.dtype());
  PrimExpr result = tir::AndNode::make(x, y);
  PrimExpr identity_element = make_const(source.dtype(), true);
  tir::CommReducer combiner =
    tir::CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return tir::ReduceNode::make(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0);
}

PrimExpr any(PrimExpr source, Array<IterVar> rdom) {
  CHECK(source.dtype().is_bool());
  Var x("x", source.dtype()), y("y", source.dtype());
  PrimExpr result = tir::OrNode::make(x, y);
  PrimExpr identity_element = make_const(source.dtype(), false);
  tir::CommReducer combiner =
    tir::CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return tir::ReduceNode::make(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0);
}

PrimExpr max(PrimExpr source, Array<IterVar> rdom) {
  Var x("x", source.dtype()), y("y", source.dtype());
  PrimExpr result = tir::MaxNode::make(x, y);
  PrimExpr identity_element = min_value(source.dtype());
  tir::CommReducer combiner =
    tir::CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return tir::ReduceNode::make(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0);
}

PrimExpr min(PrimExpr source, Array<IterVar> rdom) {
  Var x("x", source.dtype()), y("y", source.dtype());
  PrimExpr result = tir::MinNode::make(x, y);
  PrimExpr identity_element = max_value(source.dtype());
  tir::CommReducer combiner =
    tir::CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return tir::ReduceNode::make(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0);
}

PrimExpr prod(PrimExpr source, Array<IterVar> rdom) {
  Var x("x", source.dtype()), y("y", source.dtype());
  PrimExpr result = tir::MulNode::make(x, y);
  PrimExpr identity_element = make_const(source.dtype(), 1);
  tir::CommReducer combiner =
    tir::CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return tir::ReduceNode::make(combiner, {source}, rdom, make_const(DataType::Bool(1), true), 0);
}

PrimExpr fmod(PrimExpr x, PrimExpr y) {
  BinaryOpMatchTypes(x, y);
  CHECK(x.dtype().is_float()) << "fmod only applies to float";
  return tir::CallNode::make(x.dtype(), "fmod", { x, y }, tir::CallNode::PureIntrinsic);
}

PrimExpr floor(PrimExpr x) {
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::floor(fx->value));
  return tir::CallNode::make(x.dtype(), "floor", {x}, tir::CallNode::PureIntrinsic);
}

PrimExpr ceil(PrimExpr x) {
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::ceil(fx->value));
  return tir::CallNode::make(x.dtype(), "ceil", {x}, tir::CallNode::PureIntrinsic);
}

PrimExpr round(PrimExpr x) {
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::nearbyint(fx->value));
  return tir::CallNode::make(x.dtype(), "round", {x}, tir::CallNode::PureIntrinsic);
}

PrimExpr nearbyint(PrimExpr x) {
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.dtype(), std::nearbyint(fx->value));
  return tir::CallNode::make(x.dtype(), "nearbyint", {x}, tir::CallNode::PureIntrinsic);
}

PrimExpr trunc(PrimExpr x) {
  using tir::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) {
    return FloatImm(x.dtype(), (fx->value < 0 ? std::ceil(fx->value) :
                                     std::floor(fx->value)));
  }
  return tir::CallNode::make(x.dtype(), "trunc", {x}, tir::CallNode::PureIntrinsic);
}


// expose basic functions to node namespace
TVM_REGISTER_GLOBAL("node._const")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    if (args[0].type_code() == kDLInt) {
      *ret = tir::make_const(args[1], args[0].operator int64_t());
    } else if (args[0].type_code() == kDLFloat) {
      *ret = tir::make_const(args[1], args[0].operator double());
    } else {
      LOG(FATAL) << "only accept int or float";
    }
  });

TVM_REGISTER_GLOBAL("node.LargeUIntImm")
.set_body_typed(LargeUIntImm);

TVM_REGISTER_GLOBAL("node.String")
.set_body_typed(tir::StringImmNode::make);

}  // namespace tvm
