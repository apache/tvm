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
 * \file tvm/arithmetic.h
 * \brief Algebra and set operations and simplifications.
 */
#ifndef TVM_ARITHMETIC_H_
#define TVM_ARITHMETIC_H_

#include <vector>
#include <unordered_map>
#include <memory>
#include <limits>
#include "expr.h"

namespace tvm {
// forward delcare Tensor
class Tensor;
/*! \brief namespace of arithmetic */
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

/*!
 * \brief Constant integer up and lower bound(inclusive).
 *  Useful for value bound analysis.
 *
 *  set = [min_value, max_value]
 */
class ConstIntBoundNode : public Node {
 public:
  int64_t min_value;
  int64_t max_value;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("min_value", &min_value);
    v->Visit("max_value", &max_value);
  }

  /*! \brief Number to represent +inf */
  static const constexpr int64_t kPosInf = std::numeric_limits<int64_t>::max();
  /*!
   * \brief Number to represent -inf
   * \note We can make use the of fact that -kPosInf == kNegInf in the project.
   */
  static const constexpr int64_t kNegInf = -kPosInf;

  static constexpr const char* _type_key = "arith.ConstIntBound";
  TVM_DECLARE_NODE_TYPE_INFO(ConstIntBoundNode, Node);
};

/*!
 * \brief reference class to ConstIntBoundNode
 * \sa ConstIntBoundNode
 */
class ConstIntBound : public NodeRef {
 public:
  /*!
   * \brief constructor by fields.
   * \param min_value The mininum value.
   * \param max_value The maximum value.
   */
  TVM_DLL ConstIntBound(int64_t min_value, int64_t max_value);

  static const constexpr int64_t kPosInf = ConstIntBoundNode::kPosInf;
  static const constexpr int64_t kNegInf = ConstIntBoundNode::kNegInf;
  TVM_DEFINE_NODE_REF_METHODS(ConstIntBound, NodeRef, ConstIntBoundNode);
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
  ConstIntBound operator()(const Expr& expr);

  /*!
   * \brief Update constant int bound information of var.
   *
   * \param var The variable of interest.
   * \param info The bound information.
   * \param override Whether do we allow override of existing information.
   */
  void Update(const Var& var,
              const ConstIntBound& info,
              bool override = false);
  /*!
   * \brief Bind variable to a range.
   *
   * \param var The variable.
   * \param range The range we bind to.
   */
  void Bind(const Var& var, const Range& range);

 private:
  friend class Analyzer;
  friend class ConstraintContext;
  explicit ConstIntBoundAnalyzer(Analyzer* parent);
  ~ConstIntBoundAnalyzer();
  /*!
   * \brief Update the internal state to enter constraint.
   * \param constraint A constraint expression.
   *
   * \return an exit function that must be called to cleanup the constraint can be nullptr.
   */
  std::function<void()> EnterConstraint(const Expr& constraint);
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
class ModularSetNode : public Node {
 public:
  /*! \brief linear co-efficient */
  int64_t coeff;
  /*! \brief The base */
  int64_t base;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("coeff", &coeff);
    v->Visit("base", &base);
  }

  static constexpr const char* _type_key = "arith.ModularSet";
  TVM_DECLARE_NODE_TYPE_INFO(ModularSetNode, Node);
};

/*!
 * \brief reference of ModularSetNode
 * \sa ModularSetNode
 */
class ModularSet : public NodeRef {
 public:
  TVM_DLL ModularSet(int64_t coeff, int64_t base);

  TVM_DEFINE_NODE_REF_METHODS(ModularSet, NodeRef, ModularSetNode);
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
  ModularSet operator()(const Expr& expr);
  /*!
   * \brief Update constant int bound information of var.
   *
   * \param var The variable of interest.
   * \param info The bound information.
   * \param override Whether do we allow override of existing information.
   */
  void Update(const Var& var,
              const ModularSet& info,
              bool override = false);

 private:
  friend class Analyzer;
  friend class ConstraintContext;
  explicit ModularSetAnalyzer(Analyzer* parent);
  ~ModularSetAnalyzer();
  /*!
   * \brief Update the internal state to enter constraint.
   * \param constraint A constraint expression.
   *
   * \return an exit function that must be called to cleanup the constraint can be nullptr.
   */
  std::function<void()> EnterConstraint(const Expr& constraint);
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
  Expr operator()(const Expr& expr);

  /*!
   * \brief Update binding of var to a new expression.
   *
   * \param var The variable of interest.
   * \param new_expr
   * \param override Whether do we allow override of existing information.
   */
  void Update(const Var& var,
              const Expr& new_expr,
              bool override = false);

 private:
  friend class Analyzer;
  friend class ConstraintContext;
  friend class CanonicalSimplifier;
  explicit RewriteSimplifier(Analyzer* parent);
  ~RewriteSimplifier();
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
  Expr operator()(const Expr& expr);

  /*!
   * \brief Update binding of var to a new expression.
   *
   * \param var The variable of interest.
   * \param new_expr
   * \param override Whether do we allow override of existing information.
   */
  void Update(const Var& var,
              const Expr& new_expr,
              bool override = false);

 private:
  friend class Analyzer;
  friend class ConstraintContext;
  explicit CanonicalSimplifier(Analyzer* parent);
  ~CanonicalSimplifier();
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
  ConstraintContext(Analyzer* analyzer, Expr constraint)
      : analyzer_(analyzer), constraint_(constraint) {}
  // enter the scope.
  void EnterWithScope();
  // exit the scope.
  void ExitWithScope();
  /*! \brief The analyzer */
  Analyzer* analyzer_;
  /*! \brief The constraint */
  Expr constraint_;
  /*! \brief function to be called in recovery */
  std::function<void()> exit_;
};

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
 * \brief Base class of all IntSet containers.
 */
struct IntSetNode : public Node {
  static constexpr const char* _type_key = "IntSet";
  TVM_DECLARE_BASE_NODE_INFO(IntSetNode, Node);
};

/*!
 * \brief Integer set class, represent a set of integers in one dimension.
 */
class IntSet : public NodeRef {
 public:
  /*! \brief constructor */
  IntSet() {}
  // constructor from not container.
  explicit IntSet(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const IntSetNode* operator->() const;
  /*!
   * \brief Find a range that covers the region.
   * \param max_range The range to be covered.
   * \return The covering range.
   */
  Range cover_range(Range max_range) const;
  /*! \return Lower bound of the set */
  Expr min() const;
  /*! \return upper bound of the set */
  Expr max() const;
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
  Expr point_value() const;
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
  static IntSet single_point(Expr point);
  /*!
   * \brief construct a integer set from vector expression.
   * \param vec The vector expression, can also be single point.
   * \return The result set containing the indices in the vector.
   */
  static IntSet vector(Expr vec);
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
  static IntSet interval(Expr min, Expr max);
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
  IntSet operator()(const Expr& expr, const Map<Var, IntSet>& dom_map);

 private:
  friend class Analyzer;
  explicit IntSetAnalyzer(Analyzer* parent);
  ~IntSetAnalyzer();
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
class Analyzer {
 public:
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
   */
  void Bind(const VarExpr& var, const Expr& expr);
  /*!
   * \brief Notify all the sub-analyzers that var
   *        is created and binded to a range.
   *
   *  Each var can only be binded once.
   *
   * \param var The variable.
   * \param range The range we bind to.
   */
  void Bind(const VarExpr& var, const Range& range);
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
  bool CanProveGreaterEqual(const Expr& expr, int64_t lower_bound);
  /*!
   * \brief Whether can we prove condition.
   *
   * \param cond The expression to be proved.
   * \return The result.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProve(const Expr& cond);
  /*!
   * \brief Simplify expr.
   *
   * \param expr The expression to be simplified.
   * \return The result.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  Expr Simplify(const Expr& expr);
};

//-----------------------------------------------
// Integer set legacy API.
//------------------------------------------------
/*!
 * \brief Find an symbolic integer set that contains all possible values of
 *  e given the domain of each iteration variables.
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(Expr e,
               const Map<IterVar, IntSet>& dom_map);
/*!
 * \brief Same as EvalSet, but takes unordered_map
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(Expr e,
               const std::unordered_map<const Variable*, IntSet>& dom_map);

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
               const std::unordered_map<const Variable*, IntSet>& dom_map);
/*!
 * \brief Same as EvalSet, but takes unordered_map
 *
 * \param r The range to be evaluated.
 * \param dom_map The domain of each variable.
 * \return An integer set that can cover all the possible values of e.
 */
IntSet EvalSet(Range r,
               const std::unordered_map<const Variable*, IntSet>& dom_map);

/*! \brief Map from Expr to IntSet */
using ExprIntSetMap = std::unordered_map<Expr, IntSet, NodeHash, NodeEqual>;
/*!
 * \brief Find the integer set of every sub-expression, given the
 *  domain of each iteration variables.
 *
 * \param e The expression to be evaluated.
 * \param dom_map The domain of each variable.
 * \return the map from the expression to its possible value.
 */
ExprIntSetMap EvalSetForEachSubExpr(
    Expr e,
    const std::unordered_map<const Variable*, IntSet>& dom_map);

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

/*!
 * \brief Deduce the bound of the target variable in a expression,
 *  give the domain of each variables. Return undefined IntSet to
 *  represent failure.
 *
 * \note The returned set may be smaller than set that
 *       contains all possible values of v that satisfies the bound.
 *
 * \param v The target variable to be deduced.
 * \param cond The conditional expression.
 * \param hint_map The domain of variable, used to help deduce.
 * \param relax_map The domain of each variable, used to relax the domain,
 *        The deduce bound must implies e for all value in relax_map
 * \return An integer set that always satisfies the condition.
 */
IntSet DeduceBound(Expr v, Expr cond,
                   const Map<Var, IntSet>& hint_map,
                   const Map<Var, IntSet>& relax_map);
/*!
 * \brief Same as DeduceBound with  unordered_map signature.
 *
 * \param v The target variable to be deduced.
 * \param cond The conditional expression.
 * \param hint_map The domain of variable, used to help deduce.
 * \param relax_map The domain of each variable, used to relax the domain,
 *        The deduce bound mush implies e for all value in relax_map
 * \return An integer set that always satisfies the condition.
 */
IntSet DeduceBound(Expr v, Expr cond,
                   const std::unordered_map<const Variable*, IntSet>& hint_map,
                   const std::unordered_map<const Variable*, IntSet>& relax_map);

/*!
 * \brief Infer a regular domain that covers all the calls or provides within the given statement.
 * \param body The given statement.
 * \param tensor The name of the calls or provides.
 * \param consider_calls If calls (read) are considered.
 * \param consider_provides If provides (write) are considered.
 * \return The domain that covers all the calls or provides within the given statement.
 */
Domain DomainTouched(Stmt body, const Tensor &tensor, bool consider_calls, bool consider_provides);

// Expression pattern detector.
/*!
 * \brief Detect if e can be rewritten as e = sum_{i=0}^{n-1} var[i] * coeff[i] + coeff[n]
 *  Where coeff[i] and base are invariant of var[j] for all i and j.
 *
 * \param e The expression to be detected.
 * \param vars List of variables to be used in detection.
 * \return [coeff[i]] if it is possible, empty array if it is not.
 */
Array<Expr> DetectLinearEquation(const Expr& e,
                                 const Array<Var>& vars);

/*!
 * \brief Detect if expression corresponds to clip bound of the vars
 *
 * \param e The expression to be detected.
 * \param vars List of variables to be used in detection.
 * \return concat([min_value[i], max_value[i]]), None is returned if there is no min or max value
 *          return empty if the e does not match the pattern.
 */
Array<Expr> DetectClipBound(const Expr& e,
                            const Array<Var>& vars);

// implementation
inline const IntSetNode* IntSet::operator->() const {
  return static_cast<const IntSetNode*>(node_.get());
}
}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_H_
