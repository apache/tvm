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
 * \file tvm/arith/analyzer.h
 * \brief Algebra expression simplifications.
 */
#ifndef TVM_ARITH_ANALYZER_H_
#define TVM_ARITH_ANALYZER_H_

#include <tvm/support/with.h>
#include <tvm/ir/expr.h>
#include <tvm/arith/int_set.h>

#include <vector>
#include <unordered_map>
#include <memory>
#include <limits>

namespace tvm {
/*! \brief namespace of arithmetic analysis. */
namespace arith {
//-------------------------------------------------------
// Base integer analysis API.
//
// We have multiple type of analyzers to do relaxed
// integer set analysis(bound analysis, modulo) and
// equivalence checking and simplification.
//
// Importantly, each analyzer may need result from
// another analyzer.
//-------------------------------------------------------

// Forward declare Analyzer
class Analyzer;

using tir::Var;

/*!
 * \brief Constant integer up and lower bound(inclusive).
 *  Useful for value bound analysis.
 *
 *  set = [min_value, max_value]
 */
class ConstIntBoundNode : public Object {
 public:
  int64_t min_value;
  int64_t max_value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("min_value", &min_value);
    v->Visit("max_value", &max_value);
  }

  bool SEqualReduce(const ConstIntBoundNode* other, SEqualReducer equal) const {
    return equal(min_value, other->min_value) && equal(max_value, other->max_value);
  }

  /*! \brief Number to represent +inf */
  static const constexpr int64_t kPosInf = std::numeric_limits<int64_t>::max();
  /*!
   * \brief Number to represent -inf
   * \note We can make use the of fact that -kPosInf == kNegInf in the project.
   */
  static const constexpr int64_t kNegInf = -kPosInf;

  static constexpr const char* _type_key = "arith.ConstIntBound";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstIntBoundNode, Object);
};

/*!
 * \brief reference class to ConstIntBoundNode
 * \sa ConstIntBoundNode
 */
class ConstIntBound : public ObjectRef {
 public:
  /*!
   * \brief constructor by fields.
   * \param min_value The mininum value.
   * \param max_value The maximum value.
   */
  TVM_DLL ConstIntBound(int64_t min_value, int64_t max_value);

  static const constexpr int64_t kPosInf = ConstIntBoundNode::kPosInf;
  static const constexpr int64_t kNegInf = ConstIntBoundNode::kNegInf;
  TVM_DEFINE_OBJECT_REF_METHODS(ConstIntBound, ObjectRef, ConstIntBoundNode);
};

/*!
 * \brief Analyzer to get constant integer bound over expression.
 */
class ConstIntBoundAnalyzer {
 public:
  /*!
   * \brief analyze the expr
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_DLL ConstIntBound operator()(const PrimExpr& expr);

  /*!
   * \brief analyze the expr with the intermediate memorized to avoid redundant computation
   * \param expr The expression of interest.
   * \param bound The lookup table to store the intermediate results
   * \return the result of the analysis.
   */
  TVM_DLL ConstIntBound operator()(const PrimExpr& expr,
                                   std::unordered_map<const PrimExprNode*, ConstIntBound>* bound);

  /*!
   * \brief Update constant int bound information of var.
   *
   * \param var The variable of interest.
   * \param info The bound information.
   * \param override Whether do we allow override of existing information.
   */
  TVM_DLL void Update(const Var& var,
                      const ConstIntBound& info,
                      bool override = false);
  /*!
   * \brief Bind variable to a range.
   *
   * \param var The variable.
   * \param range The range we bind to.
   * \param override Whether we allow overriding an existing var's range.
   */
  TVM_DLL void Bind(const Var& var, const Range& range, bool override = false);

 private:
  friend class Analyzer;
  friend class ConstraintContext;
  explicit ConstIntBoundAnalyzer(Analyzer* parent);
  TVM_DLL ~ConstIntBoundAnalyzer();
  /*!
   * \brief Update the internal state to enter constraint.
   * \param constraint A constraint expression.
   *
   * \return an exit function that must be called to cleanup the constraint can be nullptr.
   */
  std::function<void()> EnterConstraint(const PrimExpr& constraint);
  struct Entry;
  class Impl;
  /*! \brief Internal impl */
  Impl* impl_;
};

/*!
 * \brief Range of a linear integer function.
 *  Use to do specify the possible index values.
 *
 *  set = { coeff * x + base | x in Z }
 *
 *  When coeff != 0, it can also be written as
 *  set = { n | n % coeff == base }
 *
 *  This is useful to decide if the index is dividable by certain value.
 *  For example, if index = 0 + 4 x, then we know it can be divided by 4.
 */
class ModularSetNode : public Object {
 public:
  /*! \brief linear co-efficient */
  int64_t coeff;
  /*! \brief The base */
  int64_t base;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("coeff", &coeff);
    v->Visit("base", &base);
  }

  bool SEqualReduce(const ModularSetNode* other, SEqualReducer equal) const {
    return equal(coeff, other->coeff) && equal(base, other->base);
  }

  static constexpr const char* _type_key = "arith.ModularSet";
  TVM_DECLARE_FINAL_OBJECT_INFO(ModularSetNode, Object);
};

/*!
 * \brief reference of ModularSetNode
 * \sa ModularSetNode
 */
class ModularSet : public ObjectRef {
 public:
  TVM_DLL ModularSet(int64_t coeff, int64_t base);

  TVM_DEFINE_OBJECT_REF_METHODS(ModularSet, ObjectRef, ModularSetNode);
};

/*!
 * \brief Analyzer to get modular information over expression.
 */
class ModularSetAnalyzer {
 public:
  /*!
   * \brief analyze the expr
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_DLL ModularSet operator()(const PrimExpr& expr);
  /*!
   * \brief Update constant int bound information of var.
   *
   * \param var The variable of interest.
   * \param info The bound information.
   * \param override Whether do we allow override of existing information.
   */
  TVM_DLL void Update(const Var& var,
                      const ModularSet& info,
                      bool override = false);

 private:
  friend class Analyzer;
  friend class ConstraintContext;
  explicit ModularSetAnalyzer(Analyzer* parent);
  TVM_DLL ~ModularSetAnalyzer();
  /*!
   * \brief Update the internal state to enter constraint.
   * \param constraint A constraint expression.
   *
   * \return an exit function that must be called to cleanup the constraint can be nullptr.
   */
  std::function<void()> EnterConstraint(const PrimExpr& constraint);
  struct Entry;
  class Impl;
  /*! \brief Internal impl */
  Impl* impl_;
};

/*!
 * \brief Rewrite-rule based simplifier.
 */
class RewriteSimplifier {
 public:
  /*!
   * \brief analyze the expr
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_DLL PrimExpr operator()(const PrimExpr& expr);

  /*!
   * \brief Update binding of var to a new expression.
   *
   * \param var The variable of interest.
   * \param new_expr
   * \param override Whether do we allow override of existing information.
   */
  TVM_DLL void Update(const Var& var,
                      const PrimExpr& new_expr,
                      bool override = false);

  std::function<void()> EnterConstraint(const PrimExpr& constraint);

 private:
  friend class Analyzer;
  friend class ConstraintContext;
  friend class CanonicalSimplifier;
  explicit RewriteSimplifier(Analyzer* parent);
  TVM_DLL ~RewriteSimplifier();
  class Impl;
  /*! \brief Internal impl */
  Impl* impl_;
};

/*!
 * \brief Canonical-form based simplifier.
 */
class CanonicalSimplifier {
 public:
  /*!
   * \brief analyze the expr
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_DLL PrimExpr operator()(const PrimExpr& expr);

  /*!
   * \brief Update binding of var to a new expression.
   *
   * \param var The variable of interest.
   * \param new_expr
   * \param override Whether do we allow override of existing information.
   */
  TVM_DLL void Update(const Var& var,
                      const PrimExpr& new_expr,
                      bool override = false);

 private:
  friend class Analyzer;
  friend class ConstraintContext;
  explicit CanonicalSimplifier(Analyzer* parent);
  TVM_DLL ~CanonicalSimplifier();
  class Impl;
  /*! \brief Internal impl */
  Impl* impl_;
};

/*!
 * \brief Constraint context.
 *
 * \code
 *
 *  Var("x");
 *  arith::Analyzer analyzer;
 *  {
 *    With<arith::ConstraintContext> scope(&analyzer, x % 3 == 0);
 *    CHECK_EQ(analyzer.modular_set(x)->coeff, 3);
 *  }
 *  // constraint no longer in effect.
 *  CHECK_NE(analyzer.modular_set(x)->coeff, 3);
 *
 * \endcode
 */
class ConstraintContext {
 private:
  // declare friend to enable with.
  friend class With<ConstraintContext>;
  /*!
   * \brief Construct a constraint context.
   * \param analyzer The analyzer.
   * \param constraint The constraint to be applied.
   */
  ConstraintContext(Analyzer* analyzer, PrimExpr constraint)
      : analyzer_(analyzer), constraint_(constraint) {}
  // enter the scope.
  void EnterWithScope();
  // exit the scope.
  void ExitWithScope();
  /*! \brief The analyzer */
  Analyzer* analyzer_;
  /*! \brief The constraint */
  PrimExpr constraint_;
  /*! \brief function to be called in recovery */
  std::function<void()> exit_;
};

/*!
 * \brief Integer set analyzer.
 */
class IntSetAnalyzer {
 public:
  /*!
   * \brief Find a symbolic integer set that contains all possible values of
   *        expr given the domain of each variables.
   *
   * \param expr The expression of interest.
   * \param dom_map The domain map to indicate which variable to relax.
   * \return the result of the analysis.
   */
  TVM_DLL IntSet operator()(const PrimExpr& expr, const Map<Var, IntSet>& dom_map);

 private:
  friend class Analyzer;
  explicit IntSetAnalyzer(Analyzer* parent);
  TVM_DLL ~IntSetAnalyzer();
  class Impl;
  /*! \brief Internal impl */
  Impl* impl_;
};

/*!
 * \brief Analyzer that contains bunch of sub-analyzers.
 *
 * Each sub-analyzer can make use of another sub-analyzer
 * by weak reference of this.
 *
 * NOTE for sub-analyzer developers:
 * If the analyzer uses memoization, we need to clear the internal
 * cache when information about a Var has been overridden.
 */
class TVM_DLL Analyzer {
 public:
  /*
   * Disable copy constructor.
   */
  Analyzer(const Analyzer&) = delete;
  Analyzer& operator=(const Analyzer&) = delete;
  /*! \brief sub-analyzer: const integer bound */
  ConstIntBoundAnalyzer const_int_bound;
  /*! \brief sub-analyzer: modular set */
  ModularSetAnalyzer modular_set;
  /*! \brief sub-analyzer rewrite simplify */
  RewriteSimplifier rewrite_simplify;
  /*! \brief sub-analyzer canonical simplify */
  CanonicalSimplifier canonical_simplify;
  /*! \brief sub-analyzer: int set */
  IntSetAnalyzer int_set;
  /*! \brief constructor */
  Analyzer();
  /*!
   * \brief Notify all the sub-analyzers that var
   *        is created and binded to expr.
   *
   *  Each var can only be binded once.
   *
   * \param var The variable.
   * \param expr The expression we bind to.
   * \param override Whether we allow overriding an existing var's expression.
   */
  void Bind(const Var& var, const PrimExpr& expr, bool override = false);
  /*!
   * \brief Notify all the sub-analyzers that var
   *        is created and binded to a range.
   *
   *  Each var can only be binded once.
   *
   * \param var The variable.
   * \param range The range we bind to.
   * \param override Whether we allow overriding an existing var's expression.
   */
  void Bind(const Var& var, const Range& range, bool override = false);
  /*!
   * \brief Bind all the vars in the Map
   *
   * \param variables The {variable -> range} map.
   * \param override Whether we allow overriding an existing var's expression.
   */
  void Bind(const Map<Var, Range>& variables, bool override = false);
  /*!
   * \brief Whether can we prove expr >= val.

   *  Non-negative proof is very useful in integer analysis
   *  to lower divisions and mods given difference in trunc and ceil mode.
   *
   * \param expr The expression.
   * \param lower_bound The lower bound.
   * \return Whether we can prove it.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProveGreaterEqual(const PrimExpr& expr, int64_t lower_bound);
  /*!
   * \brief Whether can we prove expr < val.

   *  Non-negative proof is very useful in integer analysis
   *  to lower divisions and mods given difference in trunc and ceil mode.
   *
   * \param expr The expression.
   * \param upper_bound The upper bound.
   * \return Whether we can prove it.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProveLess(const PrimExpr& expr, int64_t upper_bound);
  /*!
   * \brief Whether can we prove condition.
   *
   * \param cond The expression to be proved.
   * \return The result.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProve(const PrimExpr& cond);
  /*!
   * \brief Simplify expr.
   *
   * \param expr The expression to be simplified.
   * \return The result.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  PrimExpr Simplify(const PrimExpr& expr);
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_ANALYZER_H_
