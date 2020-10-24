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
 * \file const_fold.h
 * \brief Centralized location for constant folding.
 */
#ifndef TVM_ARITH_CONST_FOLD_H_
#define TVM_ARITH_CONST_FOLD_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <cmath>

#include "int_operator.h"

namespace tvm {
namespace arith {

/*!
 * \brief Try to run binary compute with constant folding.
 *
 * \param a The left operand.
 * \param b The right operand.
 * \tparam Op The operator type.
 *
 * \note a and b Must already matched data types with each other.
 * \return nullptr if constant fold fails, otherwise return folded result.
 */
template <typename Op>
inline PrimExpr TryConstFold(PrimExpr a, PrimExpr b);

/*!
 * \brief Try to run unary compute with constant folding.
 *
 * \param a The left operand.
 * \tparam Op The operator type.
 *
 * \note a and b Must already matched data types with each other.
 * \return nullptr if constant fold fails, otherwise return folded result.
 */
template <typename Op>
inline PrimExpr TryConstFold(PrimExpr a);

/*!
 * \brief Check whether type is used to represent index.
 *
 * Index types are frequently used in shape computation
 * and need to be aggressively constant-folded.
 *
 * \param type The type to represent index.
 * \return the checked result.
 */
inline bool IsIndexType(const DataType& type) {
  return type.is_int() && type.lanes() == 1 && (type.bits() == 32 || type.bits() == 64);
}

#define TVM_ARITH_CONST_PROPAGATION(BODY)        \
  using tir::FloatImmNode;                       \
  const IntImmNode* pa = a.as<IntImmNode>();     \
  const IntImmNode* pb = b.as<IntImmNode>();     \
  const FloatImmNode* fa = a.as<FloatImmNode>(); \
  const FloatImmNode* fb = b.as<FloatImmNode>(); \
  BODY;

#define TVM_INDEX_CONST_PROPAGATION(BODY)                 \
  const IntImmNode* pa = a.as<IntImmNode>();              \
  const IntImmNode* pb = b.as<IntImmNode>();              \
  const DataType& ta = a.dtype();                         \
  const DataType& tb = b.dtype();                         \
  if (arith::IsIndexType(ta) && arith::IsIndexType(tb)) { \
    BODY;                                                 \
  }

// specialization of constant folders.
template <>
inline PrimExpr TryConstFold<tir::Add>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, pa->value + pb->value);
    if (pa && pa->value == 0) return b;
    if (pb && pb->value == 0) return a;
    if (fa && fb) return FloatImm(rtype, fa->value + fb->value);
    if (fa && fa->value == 0) return b;
    if (fb && fb->value == 0) return a;
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::Sub>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, pa->value - pb->value);
    if (pb && pb->value == 0) return a;
    if (fa && fb) return FloatImm(rtype, fa->value - fb->value);
    if (fb && fb->value == 0) return a;
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::Mul>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, pa->value * pb->value);
    if (pa) {
      if (pa->value == 1) return b;
      if (pa->value == 0) return a;
    }
    if (pb) {
      if (pb->value == 1) return a;
      if (pb->value == 0) return b;
    }
    if (fa && fb) return FloatImm(rtype, fa->value * fb->value);
    if (fa) {
      if (fa->value == 1) return b;
      if (fa->value == 0) return a;
    }
    if (fb) {
      if (fb->value == 1) return a;
      if (fb->value == 0) return b;
    }
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::Div>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) {
      // due to division and mod can have different modes
      // NOTE: this will assumes truc div.
      ICHECK_NE(pb->value, 0) << "Divide by zero";
      return IntImm(rtype, pa->value / pb->value);
    }
    if (pa) {
      if (pa->value == 0) return a;
    }
    if (pb) {
      if (pb->value == 1) return a;
      ICHECK_NE(pb->value, 0) << "Divide by zero";
    }
    if (fa && fb && fb->value != 0) {
      return FloatImm(rtype, fa->value / fb->value);
    }
    if (fa && fa->value == 0) return a;
    if (fb) {
      if (fb->value == 1) return a;
      ICHECK_NE(fb->value, 0) << "Divide by zero";
    }
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::Mod>(PrimExpr a, PrimExpr b) {
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) {
      ICHECK_NE(pb->value, 0) << "Divide by zero";
      return IntImm(rtype, pa->value % pb->value);
    }
    if (pa) {
      if (pa->value == 0) return a;
    }
    if (pb) {
      if (pb->value == 1) return tir::make_zero(rtype);
      ICHECK_NE(pb->value, 0) << "Divide by zero";
    }
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::FloorDiv>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) {
      ICHECK_NE(pb->value, 0) << "Divide by zero";
      return IntImm(rtype, arith::floordiv(pa->value, pb->value));
    }
    if (pa) {
      if (pa->value == 0) return a;
    }
    if (pb) {
      if (pb->value == 1) return a;
      ICHECK_NE(pb->value, 0) << "Divide by zero";
    }
    if (fa && fb && fb->value != 0) {
      return FloatImm(rtype, std::floor(fa->value / fb->value));
    }
    if (fa && fa->value == 0) return a;
    if (fb) {
      if (fb->value == 1) return a;
      ICHECK_NE(fb->value, 0) << "Divide by zero";
    }
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::FloorMod>(PrimExpr a, PrimExpr b) {
  TVM_INDEX_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) {
      ICHECK_NE(pb->value, 0) << "Divide by zero";
      return IntImm(rtype, floormod(pa->value, pb->value));
    }
    if (pa) {
      if (pa->value == 0) return a;
    }
    if (pb) {
      if (pb->value == 1) return tir::make_zero(rtype);
      ICHECK_NE(pb->value, 0) << "Divide by zero";
    }
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::Min>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, std::min(pa->value, pb->value));
    if (fa && fb) return FloatImm(rtype, std::min(fa->value, fb->value));
  });
  if (a.same_as(b)) return a;
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::Max>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    const DataType& rtype = a.dtype();
    if (pa && pb) return IntImm(rtype, std::max(pa->value, pb->value));
    if (fa && fb) return FloatImm(rtype, std::max(fa->value, fb->value));
  });
  if (a.same_as(b)) return a;
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::GT>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    if (pa && pb) return IntImm(DataType::UInt(1), pa->value > pb->value);
    if (fa && fb) return IntImm(DataType::UInt(1), fa->value > fb->value);
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::GE>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    if (pa && pb) return IntImm(DataType::UInt(1), pa->value >= pb->value);
    if (fa && fb) return IntImm(DataType::UInt(1), fa->value >= fb->value);
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::LT>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    if (pa && pb) return IntImm(DataType::UInt(1), pa->value < pb->value);
    if (fa && fb) return IntImm(DataType::UInt(1), fa->value < fb->value);
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::LE>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    if (pa && pb) return IntImm(DataType::UInt(1), pa->value <= pb->value);
    if (fa && fb) return IntImm(DataType::UInt(1), fa->value <= fb->value);
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::EQ>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    if (pa && pb) return IntImm(DataType::UInt(1), pa->value == pb->value);
    if (fa && fb) return IntImm(DataType::UInt(1), fa->value == fb->value);
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::NE>(PrimExpr a, PrimExpr b) {
  TVM_ARITH_CONST_PROPAGATION({
    if (pa && pb) return IntImm(DataType::UInt(1), pa->value != pb->value);
    if (fa && fb) return IntImm(DataType::UInt(1), fa->value != fb->value);
  });
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::And>(PrimExpr a, PrimExpr b) {
  const IntImmNode* pa = a.as<IntImmNode>();
  const IntImmNode* pb = b.as<IntImmNode>();
  if (pa && pa->value) return b;
  if (pa && !pa->value) return a;
  if (pb && pb->value) return a;
  if (pb && !pb->value) return b;
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::Or>(PrimExpr a, PrimExpr b) {
  const IntImmNode* pa = a.as<IntImmNode>();
  const IntImmNode* pb = b.as<IntImmNode>();
  if (pa && pa->value) return a;
  if (pa && !pa->value) return b;
  if (pb && pb->value) return b;
  if (pb && !pb->value) return a;
  return PrimExpr();
}

template <>
inline PrimExpr TryConstFold<tir::Not>(PrimExpr a) {
  const IntImmNode* pa = a.as<IntImmNode>();
  if (pa) {
    return IntImm(DataType::UInt(1), !(pa->value));
  }
  return PrimExpr();
}

/*! \brief Helper namespace for symbolic value limits */
struct SymbolicLimits {
  /*! \brief positive infinity */
  static PrimExpr pos_inf_;
  /*! \brief negative infinity */
  static PrimExpr neg_inf_;
};

/*!
 * \brief Opaque expression representing positive infinity.
 *
 *  It can can only be used as parameter of by min/max
 *  for integer analysis and cannot be used in normal expressions.
 *
 * \return positive infinity.
 */
inline PrimExpr pos_inf() { return SymbolicLimits::pos_inf_; }

/*!
 * \brief Check if value is positive infinity.
 * \param value The value to be checked.
 *
 * \return The check result.
 */
inline bool is_pos_inf(const PrimExpr& value) { return value.same_as(SymbolicLimits::pos_inf_); }

/*!
 * \brief Opaque expression representing negative infinity.
 *
 *  It can can only be used as parameter of by min/max
 *  for integer analysis and cannot be used in normal expressions.
 *
 * \return negative infinity.
 */
inline PrimExpr neg_inf() { return SymbolicLimits::neg_inf_; }

/*!
 * \brief Check if value is negative infinity.
 * \param value The value to be checked.
 *
 * \return The check result.
 */
inline bool is_neg_inf(const PrimExpr& value) { return value.same_as(SymbolicLimits::neg_inf_); }

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_CONST_FOLD_H_
