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

using tir::IterVar;
using tir::Var;
using tir::VarNode;

class Analyzer;

//-----------------------------------------------
// Integer set data structure.
//
// This is a API build on top of the base
// integer analysis API to provide set analysis.
//------------------------------------------------
/*!
 * \brief Sign type of an integer expression.
 */
enum SignType { kPositive, kNegative, kZero, kUnknown };

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
  /*!
   * \brief Find a range that covers the region.
   * \param max_range The range to be covered.
   * \return The covering range.
   */
  Range CoverRange(Range max_range) const;
  /*! \return Lower bound of the set */
  PrimExpr min() const;
  /*! \return upper bound of the set */
  PrimExpr max() const;
  /*! \return The sign of the elements in the integer set */
  SignType GetSignType() const;
  /*! \return Whether the set represent nothing  */
  bool IsNothing() const;
  /*! \return Whether the set represent everything  */
  bool IsEverything() const;
  /*! \return Whether the set is a single point */
  bool IsSinglePoint() const;
  /*! \return Whether the set is proved to be bigger than 0 */
  bool CanProvePositive() const;
  /*! \return Whether the set is proved to be smaller than 0 */
  bool CanProveNegative() const;
  /*! \return Whether the set is proved to be smaller than or equal to 0 */
  bool CanProveNonPositive() const;
  /*! \return Whether the set is proved to be larger than or equal to 0 */
  bool CanProveNonNegative() const;
  /*! \return Whether the set has upper bound. */
  bool HasUpperBound() const;
  /*! \return Whether the set has lower bound. */
  bool HasLowerBound() const;

  /*!
   * \brief The single point value, call only if IsSinglePoint is true
   * \return The point value.
   */
  PrimExpr PointValue() const;
  /*!
   * \brief Try to match IntSet with range r.
   *
   * \note It is guanrateed that IntSet::FromRange(r).MatchRange(r) == true
   * \return true if we can prove they are the same.
   */
  bool MatchRange(const tvm::Range& r) const;
  /*! \return The set contains nothing */
  static IntSet Nothing();
  /*! \return The set contains everything */
  static IntSet Everything();
  /*!
   * \brief construct a point set.
   * \param point The point in the set.
   * \return construct a single point set
   */
  static IntSet SinglePoint(PrimExpr point);
  /*!
   * \brief construct a integer set from vector expression.
   * \param vec The vector expression, can also be single point.
   * \return The result set containing the indices in the vector.
   */
  static IntSet Vector(PrimExpr vec);
  /*!
   * \brief Construct a set representing a range [min, min + extent).
   * \param min The minimum of the range range
   * \param extent The extent of the range.
   * \return The constructed set.
   */
  static IntSet FromMinExtent(PrimExpr min, PrimExpr extent);
  /*!
   * \brief Construct a set representing a range.
   * \param r The range
   * \return The constructed set.
   */
  static IntSet FromRange(tvm::Range r);
  /*!
   * \brief Construct a set representing a interval.
   * \param min The minimum value of the interval.
   * \param max The maximum value of the interval.
   * \return The constructed set.
   */
  static IntSet Interval(PrimExpr min, PrimExpr max);

  TVM_DEFINE_OBJECT_REF_METHODS(IntSet, ObjectRef, IntSetNode);
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
 * \brief Find an symbolic integer set that contains all possible values of
 *  e given the domain of each variables.
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(PrimExpr e, const Map<Var, IntSet>& dom_map);
/*!
 * \brief Same as EvalSet, but takes unordered_map
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(PrimExpr e, const std::unordered_map<const tir::VarNode*, IntSet>& dom_map);
/*!
 * \brief Find an symbolic integer set that contains is union over
 *  all the possible conditional values in dom_map.
 *
 * \param r The initial range.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values.
 */
IntSet EvalSet(Range r, const Map<IterVar, IntSet>& dom_map);

/*!
 * \brief Find an symbolic integer set that contains is union over
 *  all the possible conditional values in dom_map.
 *
 * \param s The initial set.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values.
 */
IntSet EvalSet(IntSet s, const std::unordered_map<const VarNode*, IntSet>& dom_map);
/*!
 * \brief Same as EvalSet, but takes unordered_map
 *
 * \param r The range to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(Range r, const std::unordered_map<const VarNode*, IntSet>& dom_map);
/*!
 * \brief Same as EvalSet, but takes Array<Range>
 *
 * \param region The range to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An array of integer sets that can cover all the possible values.
 */
Array<IntSet> EvalSet(const Array<Range>& region, const Map<Var, IntSet>& dom_map);
/*! \brief Map from Expr to IntSet */
using ExprIntSetMap = std::unordered_map<PrimExpr, IntSet, ObjectPtrHash, ObjectPtrEqual>;
/*!
 * \brief Find the integer set of every sub-expression, given the
 *  domain of each iteration variables.
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return the map from the expression to its possible value.
 */
ExprIntSetMap EvalSetForEachSubExpr(PrimExpr e,
                                    const std::unordered_map<const VarNode*, IntSet>& dom_map);

/*!
 * \brief Create a union set of all sets, possibly relaxed
 * \param sets The sets to be combined
 * \return the set after union
 */
IntSet Union(const Array<IntSet>& sets);

/*!
 * \brief The union of N-dimensional integer sets
 * \param nd_int_sets A list of N-dimensional integer sets
 * \return An N-dimensional integer set as the result of union
 */
Array<IntSet> UnionRegion(const Array<Array<IntSet>>& nd_int_sets);

/*!
 * \brief Create a lower-bound of union set, where some of the segments may be dropped
 * \param sets The sets to be combined
 * \return the set after union
 */
IntSet UnionLowerBound(const Array<IntSet>& sets);

/*!
 * \brief The union of N-dimensional integer sets
 * \param nd_int_sets A list of N-dimensional integer sets
 * \return An N-dimensional integer set as the result of union
 */
Array<IntSet> UnionRegionLowerBound(const Array<Array<IntSet>>& nd_int_sets);

/*!
 * \brief Create an intersected set of all sets
 * \param sets The sets to be intersected
 * \return the set after intersected
 */
IntSet Intersect(const Array<IntSet>& sets);

/*!
 * \brief Converts the Ranges to IntSets
 * \param var_dom The ranges of variables
 * \return The integer sets of the variables
 */
Map<Var, arith::IntSet> AsIntSet(const Map<Var, Range>& var_dom);

/*!
 * \brief Analyze the region with affine map, given the domain of variables and their predicate.
 * The result should be strict, i.e. no region is discarded or relaxed.
 * \param region The region to be analyzed
 * \param var_dom The ranges of the variables
 * \param predicate The predicate for the affine map
 * \param analyzer The analyzer used
 * \return NullOpt if the detection fails, or an array of arith::IntSet as the result of analysis
 */
TVM_DLL Optional<Array<IntSet>> EstimateRegionStrictBound(const Array<Range>& region,
                                                          const Map<Var, Range>& var_dom,
                                                          const PrimExpr& predicate,
                                                          arith::Analyzer* analyzer);

/*!
 * \brief Analyze the region with affine map, given the domain of variables and their predicate.
 * Some subregion may be discarded during the lower-bound analysis.
 * \param region The region to be analyzed
 * \param var_dom The ranges of the variables
 * \param predicate The predicate for the affine map
 * \param analyzer The analyzer used
 * \return NullOpt if the detection fails, or an array of arith::IntSet as the result of analysis
 */
TVM_DLL Optional<Array<IntSet>> EstimateRegionLowerBound(const Array<Range>& region,
                                                         const Map<Var, Range>& var_dom,
                                                         const PrimExpr& predicate,
                                                         arith::Analyzer* analyzer);

/*!
 * \brief Analyze the region with affine map, given the domain of variables and their predicate
 * Relaxation of the region may be used in upper-bound analysis, i.e. some extra region may be added
 * to the result.
 * \param region The region to be analyzed
 * \param var_dom The ranges of the variables
 * \param predicate The predicate for the affine map
 * \param analyzer The analyzer used
 * \return an array of arith::IntSet as the result of analysis
 */
TVM_DLL Array<IntSet> EstimateRegionUpperBound(const Array<Range>& region,
                                               const Map<Var, Range>& var_dom,
                                               const PrimExpr& predicate,
                                               arith::Analyzer* analyzer);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_INT_SET_H_
