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
#ifndef TVM_ARITHMETIC_CONST_FOLD_H_
#define TVM_ARITHMETIC_CONST_FOLD_H_

#include <tvm/ir.h>
#include <tvm/expr_operator.h>
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
template<typename Op>
inline Expr TryConstFold(Expr a, Expr b) {
  return Expr();
}

/*!
 * \brief Try to run unary compute with constant folding.
 *
 * \param a The left operand.
 * \tparam Op The operator type.
 *
 * \note a and b Must already matched data types with each other.
 * \return nullptr if constant fold fails, otherwise return folded result.
 */
template<typename Op>
inline Expr TryConstFold(Expr a);

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
  return type.is_int() && type.lanes() == 1 &&
      (type.bits() == 32 || type.bits() == 64);
}


#define TVM_ARITH_CONST_PROPAGATION(BODY)                               \
  using ir::IntImmNode;                                                 \
  using ir::UIntImmNode;                                                \
  using ir::FloatImmNode;                                               \
  const IntImmNode* pa = a.as<IntImmNode>();                            \
  const IntImmNode* pb = b.as<IntImmNode>();                            \
  const FloatImmNode* fa = a.as<FloatImmNode>();                        \
  const FloatImmNode* fb = b.as<FloatImmNode>();                        \
  BODY;


#define TVM_INDEX_CONST_PROPAGATION(BODY)                               \
  using ir::IntImmNode;                                                 \
  using ir::UIntImmNode;                                                \
  const IntImmNode* pa = a.as<IntImmNode>();                            \
  const IntImmNode* pb = b.as<IntImmNode>();                            \
  const DataType& ta = a.dtype();                                       \
  const DataType& tb = b.dtype();                                       \
  if (arith::IsIndexType(ta) && arith::IsIndexType(tb)) {               \
    BODY;                                                               \
  }                                                                     \


// specialization of constant folders.
template<>
inline Expr TryConstFold<ir::AddNode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) return IntImmNode::make(rtype, pa->value + pb->value);
      if (pa && pa->value == 0) return b;
      if (pb && pb->value == 0) return a;
      if (fa && fb) return FloatImmNode::make(rtype, fa->value + fb->value);
      if (fa && fa->value == 0) return b;
      if (fb && fb->value == 0) return a;
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::SubNode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) return IntImmNode::make(rtype, pa->value - pb->value);
      if (pb && pb->value == 0) return a;
      if (fa && fb) return FloatImmNode::make(rtype, fa->value - fb->value);
      if (fb && fb->value == 0) return a;
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::MulNode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) return IntImmNode::make(rtype, pa->value * pb->value);
      if (pa) {
        if (pa->value == 1) return b;
        if (pa->value == 0) return a;
      }
      if (pb) {
        if (pb->value == 1) return a;
        if (pb->value == 0) return b;
      }
      if (fa && fb) return FloatImmNode::make(rtype, fa->value * fb->value);
      if (fa) {
        if (fa->value == 1) return b;
        if (fa->value == 0) return a;
      }
      if (fb) {
        if (fb->value == 1) return a;
        if (fb->value == 0) return b;
      }
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::DivNode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) {
        // due to division and mod can have different modes
        // NOTE: this will assumes truc div.
        CHECK_NE(pb->value, 0) << "Divide by zero";
        return IntImmNode::make(rtype, pa->value / pb->value);
      }
      if (pa) {
        if (pa->value == 0) return a;
      }
      if (pb) {
        if (pb->value == 1) return a;
        CHECK_NE(pb->value, 0) << "Divide by zero";
      }
      if (fa && fb && fb->value != 0) {
        return FloatImmNode::make(rtype, fa->value / fb->value);
      }
      if (fa && fa->value == 0) return a;
      if (fb) {
        if (fb->value == 1) return a;
        CHECK_NE(fb->value, 0) << "Divide by zero";
      }
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::ModNode>(Expr a, Expr b) {
  TVM_INDEX_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) {
        return IntImmNode::make(rtype, pa->value % pb->value);
      }
      if (pa) {
        if (pa->value == 0) return a;
      }
      if (pb) {
        if (pb->value == 1) return make_zero(rtype);
        CHECK_NE(pb->value, 0) << "Divide by zero";
      }
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::FloorDivNode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) {
        CHECK_NE(pb->value, 0) << "Divide by zero";
        return IntImmNode::make(rtype, arith::floordiv(pa->value, pb->value));
      }
      if (pa) {
        if (pa->value == 0) return a;
      }
      if (pb) {
        if (pb->value == 1) return a;
        CHECK_NE(pb->value, 0) << "Divide by zero";
      }
      if (fa && fb && fb->value != 0) {
        return FloatImmNode::make(rtype, std::floor(fa->value / fb->value));
      }
      if (fa && fa->value == 0) return a;
      if (fb) {
        if (fb->value == 1) return a;
        CHECK_NE(fb->value, 0) << "Divide by zero";
      }
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::FloorModNode>(Expr a, Expr b) {
  TVM_INDEX_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) {
        return IntImmNode::make(rtype, arith::floormod(pa->value, pb->value));
      }
      if (pa) {
        if (pa->value == 0) return a;
      }
      if (pb) {
        if (pb->value == 1) return make_zero(rtype);
        CHECK_NE(pb->value, 0) << "Divide by zero";
      }
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::MinNode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) return IntImmNode::make(rtype, std::min(pa->value, pb->value));
      if (fa && fb) return FloatImmNode::make(rtype, std::min(fa->value, fb->value));
    });
  if (a.same_as(b)) return a;
  return Expr();
}

template<>
inline Expr TryConstFold<ir::MaxNode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      const DataType& rtype = a.dtype();
      if (pa && pb) return IntImmNode::make(rtype, std::max(pa->value, pb->value));
      if (fa && fb) return FloatImmNode::make(rtype, std::max(fa->value, fb->value));
    });
  if (a.same_as(b)) return a;
  return Expr();
}

template<>
inline Expr TryConstFold<ir::GTNode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImmNode::make(DataType::UInt(1), pa->value > pb->value);
      if (fa && fb) return UIntImmNode::make(DataType::UInt(1), fa->value > fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::GENode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImmNode::make(DataType::UInt(1), pa->value >= pb->value);
      if (fa && fb) return UIntImmNode::make(DataType::UInt(1), fa->value >= fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::LTNode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImmNode::make(DataType::UInt(1), pa->value < pb->value);
      if (fa && fb) return UIntImmNode::make(DataType::UInt(1), fa->value < fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::LENode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImmNode::make(DataType::UInt(1), pa->value <= pb->value);
      if (fa && fb) return UIntImmNode::make(DataType::UInt(1), fa->value <= fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::EQNode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImmNode::make(DataType::UInt(1), pa->value == pb->value);
      if (fa && fb) return UIntImmNode::make(DataType::UInt(1), fa->value == fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::NENode>(Expr a, Expr b) {
  TVM_ARITH_CONST_PROPAGATION({
      if (pa && pb) return UIntImmNode::make(DataType::UInt(1), pa->value != pb->value);
      if (fa && fb) return UIntImmNode::make(DataType::UInt(1), fa->value != fb->value);
    });
  return Expr();
}

template<>
inline Expr TryConstFold<ir::AndNode>(Expr a, Expr b) {
  using ir::UIntImmNode;
  const UIntImmNode* pa = a.as<UIntImmNode>();
  const UIntImmNode* pb = b.as<UIntImmNode>();
  if (pa && pa->value) return b;
  if (pa && !pa->value) return a;
  if (pb && pb->value) return a;
  if (pb && !pb->value) return b;
  return Expr();
}

template<>
inline Expr TryConstFold<ir::OrNode>(Expr a, Expr b) {
  using ir::UIntImmNode;
  const UIntImmNode* pa = a.as<UIntImmNode>();
  const UIntImmNode* pb = b.as<UIntImmNode>();
  if (pa && pa->value) return a;
  if (pa && !pa->value) return b;
  if (pb && pb->value) return b;
  if (pb && !pb->value) return a;
  return Expr();
}

template<>
inline Expr TryConstFold<ir::NotNode>(Expr a) {
  using ir::UIntImmNode;
  const UIntImmNode* pa = a.as<UIntImmNode>();
  if (pa) {
    return UIntImmNode::make(DataType::UInt(1), !(pa->value));
  }
  return Expr();
}

/*! \brief Helper namespace for symbolic value limits */
struct SymbolicLimits {
  /*! \brief positive infinity */
  static Expr pos_inf_;
  /*! \brief negative infinity */
  static Expr neg_inf_;
};

/*!
 * \brief Opaque expression representing positive infinity.
 *
 *  It can can only be used as parameter of by min/max
 *  for integer analysis and cannot be used in normal expressions.
 *
 * \return positive infinity.
 */
inline Expr pos_inf() {
  return SymbolicLimits::pos_inf_;
}

/*!
 * \brief Check if value is positive infinity.
 * \param value The value to be checked.
 *
 * \return The check result.
 */
inline bool is_pos_inf(const Expr& value) {
  return value.same_as(SymbolicLimits::pos_inf_);
}

/*!
 * \brief Opaque expression representing negative infinity.
 *
 *  It can can only be used as parameter of by min/max
 *  for integer analysis and cannot be used in normal expressions.
 *
 * \return negative infinity.
 */
inline Expr neg_inf() {
  return SymbolicLimits::neg_inf_;
}

/*!
 * \brief Check if value is negative infinity.
 * \param value The value to be checked.
 *
 * \return The check result.
 */
inline bool is_neg_inf(const Expr& value) {
  return value.same_as(SymbolicLimits::neg_inf_);
}

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_CONST_FOLD_H_
