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
 * \file int_set.h
 * \brief Internal data structure for integer set.
 */
#ifndef TVM_ARITH_INTERVAL_SET_H_
#define TVM_ARITH_INTERVAL_SET_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>

#include <limits>

#include "const_fold.h"

namespace tvm {
namespace arith {

// Acknowledgement: IntervalSet design originates from Halide.
/*!
 * \brief Symbolic interval set.
 *
 * \note We intentionally keep the internal of IntSet private,
         as we might change it later.
 */
class IntervalSetNode : public IntSetNode {
 public:
  /*! \brief Minimum value in the interval. */
  PrimExpr min_value;
  /*! \brief Maximum value in the interval. */
  PrimExpr max_value;

  // visitor overload.
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("min_value", &min_value);
    v->Visit("max_value", &max_value);
  }

  /*! \return Whether the interval has upper bound. */
  bool HasUpperBound() const { return !is_pos_inf(max_value) && !IsEmpty(); }
  /*! \return Whether the interval has lower bound. */
  bool HasLowerBound() const { return !is_neg_inf(min_value) && !IsEmpty(); }
  /*! \return Whether the interval is a single point. */
  bool IsSinglePoint() const {
    // NOTE: we are only doing cheap check as this is a frequently called routine,
    // do manual prove of min and max for stronger single point check.
    return min_value.same_as(max_value);
  }

  /*! \return whether interval represent nothing */
  bool IsEmpty() const {
    // during computations, either extreme could occur.
    return is_pos_inf(min_value) || is_neg_inf(max_value);
  }
  /*! \return whether interval represent everything */
  bool IsEverything() const { return is_neg_inf(min_value) && is_pos_inf(max_value); }

  static constexpr const char* _type_key = "arith.IntervalSet";
  TVM_DECLARE_FINAL_OBJECT_INFO(IntervalSetNode, IntSetNode);
};

/*!
 * \brief Interval set used for symbolic integer analysis.
 * \sa IntervalSetNode
 */
class IntervalSet : public IntSet {
 public:
  /*!
   * \brief Make a new instance of interval set.
   * \param min_value The minimum value in the interval.
   * \param max_value The maximum value in the interval.
   * \return The created set.
   */
  TVM_DLL IntervalSet(PrimExpr min_value, PrimExpr max_value);

  /*!
   * \brief Create an IntervalSet that represents a single point.
   * \param value The value to be represented.
   * \return The result set.
   */
  static IntervalSet SinglePoint(PrimExpr value) { return IntervalSet(value, value); }
  /*!
   * \brief Create an IntervalSet that represents everything.
   * \param value The value to be represented.
   * \return The result set.
   */
  static IntervalSet Everything() { return IntervalSet(neg_inf(), pos_inf()); }
  /*!
   * \brief Create an empty eet.
   * \return The result set.
   */
  static IntervalSet Empty() { return IntervalSet(pos_inf(), neg_inf()); }

  TVM_DEFINE_OBJECT_REF_COW_METHOD(IntervalSetNode);
  TVM_DEFINE_OBJECT_REF_METHODS(IntervalSet, IntSet, IntervalSetNode);
};

/*!
 * \brief Create union of two IntervalSets.
 * \param analyzer The analyzer for simplification analysis.
 * \param a The first set.
 * \param b The second set.
 * \return The result set.
 */
TVM_DLL IntervalSet Union(Analyzer* analyzer, IntervalSet a, IntervalSet b);

/*!
 * \brief Create insersection of two IntervalSets.
 * \param analzyer The analyzer for simplification analysis.
 * \param a The first set.
 * \param b The second set.
 * \return The result set.
 */
TVM_DLL IntervalSet Intersect(Analyzer* analzyer, IntervalSet a, IntervalSet b);

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_INTERVAL_SET_H_
