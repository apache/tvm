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
 * \file presburger_set.h
 * \brief Integer set based on MLIR Presburger set
 */
#ifndef TVM_ARITH_PRESBURGER_SET_H_
#define TVM_ARITH_PRESBURGER_SET_H_

#ifdef TVM_MLIR_VERSION
#if TVM_MLIR_VERSION >= 150
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/Analysis/Presburger/PresburgerRelation.h>
#include <mlir/Analysis/Presburger/Simplex.h>
#endif
#endif

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>

#include <limits>
#include <vector>

#include "const_fold.h"

namespace tvm {
namespace arith {

#ifdef TVM_MLIR_VERSION
#if TVM_MLIR_VERSION >= 150
using namespace mlir;
using namespace presburger;

// Acknowledgement: PresburgerSet is based on Presburger set of MLIR.
/*!
 * \brief Symbolic integer set.
 *
 * \note PresburgerSet aims to provide compatible APIs with IntSet,
 *       and some additional APIs that analyze and solve
 *       multi-dimension interger set problems
 */
class PresburgerSetNode : public IntSetNode {
 public:
  PresburgerSetNode() : space(PresburgerSpace::getRelationSpace()) {}
  explicit PresburgerSetNode(const PresburgerSpace& space, const Array<Var>& vars)
      : disjuncts({}), space(space), vars(vars) {}
  explicit PresburgerSetNode(const std::vector<IntegerRelation>& disjuncts,
                             const PresburgerSpace& space, const Array<Var>& vars)
      : disjuncts(disjuncts), space(space), vars(vars) {}

  /*! \brief Represent the union of multiple IntegerRelation */
  std::vector<IntegerRelation> disjuncts;
  /*! \brief The space of all the disjuncts */
  PresburgerSpace space;

  // visitor overload.
  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*!
   * \brief Do inplace union with given disjunct
   * \param disjunct The given disjunct to be union with
   */
  void unionInPlace(const IntegerRelation& disjunct) {
    assert(space.isCompatible(disjunct.getSpace()) && "Spaces should match");
    disjuncts.push_back(disjunct);
  }

  /*!
   * \brief Update int set with given constraint
   * \param constraint The added constraint to the PresburgerSet.
   * \param vars The specified domain vars in constraint expression.
   */
  void UpdateConstraint(const PrimExpr& constraint, const Array<Var>& vars);

  /*!
   * \brief Generate expression that represents the constraint
   * \return The generated expression
   */
  PrimExpr GenerateConstraint() const;

  /*!
   * \brief Set domain vars
   * \param new_vars Vars that will be taken as the domain vars
   */
  void SetVars(const Array<Var>& new_vars) { vars = new_vars; }

  /*!
   * \brief Get the current domain vars
   * \return The current doamin vars
   */
  Array<Var> GetVars() const { return vars; }

  /*! \return whether integer set is empty */
  bool IsEmpty() const {
    return std::all_of(disjuncts.begin(), disjuncts.end(),
                       std::mem_fn(&IntegerRelation::isIntegerEmpty));
  }

  static constexpr const char* _type_key = "arith.PresburgerSet";
  TVM_DECLARE_FINAL_OBJECT_INFO(PresburgerSetNode, IntSetNode);

 private:
  Array<Var> vars;
};

/*!
 * \brief Integer set used for multi-dimension integer analysis.
 * \sa PresburgerSetNode
 */
class PresburgerSet : public IntSet {
 public:
  /*!
   * \brief Make a new instance of PresburgerSet.
   * \param disjuncts The disjunts to construct the set.
   * \param vars The variables that the constraint describes about.
   * \return The created PresburgerSet.
   */
  TVM_DLL PresburgerSet(const std::vector<IntegerRelation>& disjuncts, const Array<Var>& vars);

  /*!
   * \brief Make a new instance of PresburgerSet, collect all vars as space vars.
   * \param constraint The constraint to construct the set.
   * \return The created PresburgerSet.
   */
  TVM_DLL PresburgerSet(const PrimExpr& constraint);

  TVM_DEFINE_OBJECT_REF_COW_METHOD(PresburgerSetNode);
  TVM_DEFINE_OBJECT_REF_METHODS(PresburgerSet, IntSet, PresburgerSetNode);
};
#endif  // TVM_MLIR_VERSION >= 150
#else   // TVM_MLIR_VERSION
// Class definition when MLIR is not enabled, to prevent compile-time error.
class PresburgerSetNode : public IntSetNode {
 public:
  // dummy visitor overload.
  void VisitAttrs(tvm::AttrVisitor* v) { LOG(FATAL) << "MLIR is not enabled!"; }

  static constexpr const char* _type_key = "arith.PresburgerSet";
  TVM_DECLARE_FINAL_OBJECT_INFO(PresburgerSetNode, IntSetNode);
};

class PresburgerSet : public IntSet {
 public:
  /*!
   * \brief Constructor interface to prompt when MLIR is not enabled.
   * \param constraint The constraint to construct the set.
   * \return The created set.
   */
  TVM_DLL PresburgerSet(const PrimExpr& constraint) { LOG(FATAL) << "MLIR is not enabled!"; }
};
#endif  // TVM_MLIR_VERSION
/*!
 * \brief Create a union set of all sets
 * \param sets The sets to be combined
 * \return the set after union
 */
PresburgerSet Union(const Array<PresburgerSet>& sets);

/*!
 * \brief Create an intersected set of all sets
 * \param sets The sets to be intersected
 * \return The intersect set
 */
PresburgerSet Intersect(const Array<PresburgerSet>& sets);

/*!
 * \brief Evaluate the range of given expression based on the constraint
 * in given PresburgerSet
 * \param e The target expresision to be evaluated.
 * \param set The PresburgerSet defining the constraint.
 */
IntSet EvalSet(const PrimExpr& e, const PresburgerSet& set);

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_PRESBURGER_SET_H_
