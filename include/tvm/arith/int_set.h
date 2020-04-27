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
 * \file tvm/arith/int_set.h
 * \brief Integer set
 */
#ifndef TVM_ARITH_INT_SET_H_
#define TVM_ARITH_INT_SET_H_

#include <tvm/ir/expr.h>
#include <tvm/tir/expr.h>
#include <unordered_map>

namespace tvm {
namespace arith {

using tir::Var;
using tir::VarNode;
using tir::IterVar;

//-----------------------------------------------
// Integer set data structure.
//
// This is a API build on top of the base
// integer analysis API to provide set analysis.
//------------------------------------------------
/*!
 * \brief Sign type of an integer expression.
 */
enum SignType {
  kPositive,
  kNegative,
  kZero,
  kUnknown
};

/*!
 * \brief Base class of all Integer set containers.
 *        represent a set of integers in one dimension.
 * \sa IntSet
 */
class IntSetNode : public Object {
 public:
  static constexpr const char* _type_key = "IntSet";
  static constexpr bool _type_has_method_sequal_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(IntSetNode, Object);
};

/*!
 * \brief Managed reference to IntSetNode.
 * \sa IntSetNode
 */
class IntSet : public ObjectRef {
 public:
  /*! \brief constructor */
  IntSet() {}
  // constructor from not container.
  explicit IntSet(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  const IntSetNode* operator->() const {
    return static_cast<const IntSetNode*>(get());
  }
  /*!
   * \brief Find a range that covers the region.
   * \param max_range The range to be covered.
   * \return The covering range.
   */
  Range cover_range(Range max_range) const;
  /*! \return Lower bound of the set */
  PrimExpr min() const;
  /*! \return upper bound of the set */
  PrimExpr max() const;
  /*! \return Whether the set represent nothing  */
  bool is_nothing() const;
  /*! \return Whether the set represent everything  */
  bool is_everything() const;
  /*! \return Whether the set is a single point */
  bool is_single_point() const;
  /*! \return Whether the set is proved to be bigger than 0 */
  bool can_prove_positive() const;
  /*! \return Whether the set is proved to be smaller than 0 */
  bool can_prove_negative() const;
  /*! \return Whether the set is proved to be smaller than or equal to 0 */
  bool can_prove_non_positive() const;
  /*! \return Whether the set is proved to be larger than or equal to 0 */
  bool can_prove_non_negative() const;
  /*! \return The sign of the elements in the integer set */
  SignType sign_type() const;
  /*!
   * \brief The single point value, call only if is_single_point is true
   * \return The point value.
   */
  PrimExpr point_value() const;
  /*!
   * \brief Try to match IntSet with range r.
   *
   * \note It is guanrateed that IntSet::range(r).match_range(r) == true
   * \return true if we can prove they are the same.
   */
  bool match_range(const Range& r) const;
  /*! \return The set contains nothing */
  static IntSet nothing();
  /*! \return The set contains everything */
  static IntSet everything();
  /*!
   * \brief construct a point set.
   * \param point The point in the set.
   * \return construct a single point set
   */
  static IntSet single_point(PrimExpr point);
  /*!
   * \brief construct a integer set from vector expression.
   * \param vec The vector expression, can also be single point.
   * \return The result set containing the indices in the vector.
   */
  static IntSet vector(PrimExpr vec);
  /*!
   * \brief Construct a set representing a range.
   * \param r The range
   * \return constructed set.
   */
  static IntSet range(Range r);
  /*!
   * \brief Construct a set representing a interval.
   * \param min The minimum value of the interval.
   * \param max The maximum value of the interval.
   * \return constructed set.
   */
  static IntSet interval(PrimExpr min, PrimExpr max);
};

//-----------------------------------------------
// Integer set legacy API.
//------------------------------------------------
/*!
 * \brief Convert std::unordered_map<const VarNode*, IntSet> to Map<Var, IntSet>
 *
 * \param dom_map The domain map to convert.
 * \return The converted map.
 */
Map<Var, IntSet> ConvertDomMap(const std::unordered_map<const VarNode*, IntSet>& dom_map);
/*!
 * \brief Find an symbolic integer set that contains all possible values of
 *  e given the domain of each iteration variables.
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(PrimExpr e, const Map<IterVar, IntSet>& dom_map);
/*!
 * \brief Same as EvalSet, but takes unordered_map
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(PrimExpr e,
               const std::unordered_map<const tir::VarNode*, IntSet>& dom_map);
/*!
 * \brief Find an symbolic integer set that contains is union over
 *  all the possible conditional values in dom_map.
 *
 * \param r The initial range.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values.
 */
IntSet EvalSet(Range r,
               const Map<IterVar, IntSet>& dom_map);

/*!
 * \brief Find an symbolic integer set that contains is union over
 *  all the possible conditional values in dom_map.
 *
 * \param s The initial set.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values.
 */
IntSet EvalSet(IntSet s,
               const std::unordered_map<const VarNode*, IntSet>& dom_map);
/*!
 * \brief Same as EvalSet, but takes unordered_map
 *
 * \param r The range to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(Range r,
               const std::unordered_map<const VarNode*, IntSet>& dom_map);
/*! \brief Map from Expr to IntSet */
using ExprIntSetMap = std::unordered_map<PrimExpr, IntSet, ObjectHash, ObjectEqual>;
/*!
 * \brief Find the integer set of every sub-expression, given the
 *  domain of each iteration variables.
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return the map from the expression to its possible value.
 */
ExprIntSetMap EvalSetForEachSubExpr(
    PrimExpr e,
    const std::unordered_map<const VarNode*, IntSet>& dom_map);

/*!
 * \brief Create an union set of all sets
 * \param sets The sets to be unioned
 * \return the set after union
 */
IntSet Union(const Array<IntSet>& sets);

/*!
 * \brief Create an union set of all sets
 * \param sets The sets to be intersected
 * \return the set after intersected
 */
IntSet Intersect(const Array<IntSet>& sets);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_INT_SET_H_
