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

#include <tvm/arith/int_set.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/with_context.h>

#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

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

// Forward declare the analyzer object and its reference handle.
class AnalyzerObj;
class Analyzer;
class ConstraintContext;

using tirx::Var;

enum DivMode {
  /*! \brief Truncated division. */
  kTruncDiv,
  /*! \brief Floor division. */
  kFloorDiv
};

/*!
 * \brief The strength used in top-level condition proves
 * \note The higher, the more time consuming it can be.
 *
 * Do not use level beyond kDefault in internal recursive rewriting in arith
 * analysis and only use it at top-level simplification to avoid speed issues.
 */
enum class ProofStrength : int {
  /*! \brief default strength, can be used in. */
  kDefault = 0,
  /*!
   * \brief Prove using symbolic bound analysis
   */
  kSymbolicBound = 1,
};

/*!
 * \brief Constant integer up and lower bound(inclusive).
 *  Useful for value bound analysis.
 *
 *  set = [min_value, max_value]
 */
class ConstIntBoundNode : public ffi::Object {
 public:
  int64_t min_value;
  int64_t max_value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ConstIntBoundNode>()
        .def_ro("min_value", &ConstIntBoundNode::min_value)
        .def_ro("max_value", &ConstIntBoundNode::max_value);
  }

  /*! \brief Number to represent +inf */
  static const constexpr int64_t kPosInf = std::numeric_limits<int64_t>::max();
  /*!
   * \brief Number to represent -inf
   * \note We can make use the of fact that -kPosInf == kNegInf in the project.
   */
  static const constexpr int64_t kNegInf = -kPosInf;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("arith.ConstIntBound", ConstIntBoundNode, ffi::Object);
};

/*!
 * \brief reference class to ConstIntBoundNode
 * \sa ConstIntBoundNode
 */
class ConstIntBound : public ffi::ObjectRef {
 public:
  /*!
   * \brief constructor by fields.
   * \param min_value The mininum value.
   * \param max_value The maximum value.
   */
  TVM_DLL ConstIntBound(int64_t min_value, int64_t max_value);

  static const constexpr int64_t kPosInf = ConstIntBoundNode::kPosInf;
  static const constexpr int64_t kNegInf = ConstIntBoundNode::kNegInf;
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ConstIntBound, ffi::ObjectRef, ConstIntBoundNode);
};

/*!
 * \brief Analyzer to get constant integer bound over expression.
 */
class ConstIntBoundAnalyzer {
 public:
  using BoundMapType =
      std::unordered_map<PrimExpr, ConstIntBound, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
  /*!
   * \brief analyze the expr
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_DLL ConstIntBound operator()(const PrimExpr& expr) const;

  /*!
   * \brief analyze the expr with the intermediate memorized to avoid redundant computation
   * \param expr The expression of interest.
   * \param bound The lookup table to store the intermediate results
   * \return the result of the analysis.
   */
  TVM_DLL ConstIntBound operator()(const PrimExpr& expr, BoundMapType* bound);

  /*!
   * \brief Update constant int bound information of var.
   *
   * \param var The variable of interest.
   * \param info The bound information.
   * \param allow_override whether we allow override of existing information.
   */
  TVM_DLL void Update(const Var& var, const ConstIntBound& info, bool allow_override = false);

  /*!
   * \brief Bind variable to a range.
   *
   * \param var The variable.
   * \param range The range we bind to.
   * \param allow_override Whether we allow overriding an existing var's range.
   */
  TVM_DLL void Bind(const Var& var, const Range& range, bool allow_override = false);

  /*!
   * \brief Check if a variable is bound to a range.
   * \param var The variable.
   * \return Whether the variable is bound to a range.
   */
  TVM_DLL bool IsBound(const Var& var) const;

 private:
  friend class AnalyzerObj;
  friend class ConstraintContext;
  explicit ConstIntBoundAnalyzer(AnalyzerObj* parent);
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
class ModularSetNode : public ffi::Object {
 public:
  /*! \brief linear co-efficient */
  int64_t coeff;
  /*! \brief The base */
  int64_t base;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ModularSetNode>()
        .def_ro("coeff", &ModularSetNode::coeff)
        .def_ro("base", &ModularSetNode::base);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("arith.ModularSet", ModularSetNode, ffi::Object);
};

/*!
 * \brief reference of ModularSetNode
 * \sa ModularSetNode
 */
class ModularSet : public ffi::ObjectRef {
 public:
  TVM_DLL ModularSet(int64_t coeff, int64_t base);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ModularSet, ffi::ObjectRef, ModularSetNode);
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
   * \param allow_override whether we allow override of existing information.
   */
  TVM_DLL void Update(const Var& var, const ModularSet& info, bool allow_override = false);

 private:
  friend class AnalyzerObj;
  friend class ConstraintContext;
  explicit ModularSetAnalyzer(AnalyzerObj* parent);
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
   * \param allow_override Whether we allow override of existing information.
   */
  TVM_DLL void Update(const Var& var, const PrimExpr& new_expr, bool allow_override = false);

  /*!
   * \brief Update the internal state to enter constraint.
   * \param constraint A constraint expression.
   *
   * \return an exit function that must be called to cleanup the constraint can be nullptr.
   */
  TVM_DLL std::function<void()> EnterConstraint(const PrimExpr& constraint);

  /*! \brief Flags to enable more computationally-intensive simplifications
   *
   * These simplifications may be required for specific schedules, but
   * would impose too high a compile-time cost to enable by default.
   * They can be enabled on an as-needed basis by calling
   * `RewriteSimplifier::SetEnabledExtensions` prior to using
   * `RewriteSimplifier::operator()`.
   *
   * Flags are defined as powers of two to allow future expansion.  To
   * enable multiple extensions, a user should pass a bitwise OR of the
   * flags for each desired extension.
   */
  enum Extension {
    // No extensions enabled
    kNone = 0,

    /* When simplifying an inequality, attempt to use scope-based knowns.
     *
     * Example:
     * if_then_else(i<j && j<k, i<k, false) => if_then_else(i<j && j<k, true, false)
     */
    kTransitivelyProveInequalities = (1 << 0),

    /* When simplifying a boolean expression, convert to an AND of ORs
     * (conjunctive normal form).
     *
     * Example:
     *   (a && b) || c => (a || c) && (b || c)
     */
    kConvertBooleanToAndOfOrs = (1 << 1),

    /* When simplifying a boolean AND or a boolean OR, simplify each
     * branch under the assumption that the other branch does not
     * already dominate the result.  That is, simplify each branch of
     * (A && B) under the assumption that the other branch is true,
     * and simplify each branch of (A || B) under the assumption that
     * the other branch is false.
     *
     * Example:
     *   (n < 10) && (n < 5) => (n < 10)
     *   (n < 10) || (n < 5) => (n < 5)
     */
    kApplyConstraintsToBooleanBranches = (1 << 2),

    /* Special handling for expressions `(A+B)*C < (A*B)*D`
     *
     * Expressions of the form `(A+B)*C < (A*B)*D` can occur occur
     * when comparing the number of operations required for two
     * different orderings in which matrix multiplications can be
     * performed.  Proving or disproving this conditional allows an
     * optimal order of execution to be selected, even for dynamic
     * argument shapes.
     *
     * The default behavior of `ConstIntBounds` assumes that each term
     * in an expression is independent, and is insufficient to prove
     * these inequalities.  For example, the maximum value of `(A+B)*C
     * - (A*B)*D` is determined by taking the maximum value of
     * `(A+B)*C` and subtracting the minimum value of `(A*B)*D`.
     * While this algorithm can be applied in all cases, the bound it
     * provides is looser than strictly required.
     *
     * This extension adds a check for this case.  When `A`, `B`, `C`,
     * and `D` are all positive values, as is the case for tensor
     * shapes, the inequality can be written as `1/A + 1/B < D/C`.  If
     * this inequality holds for the minimum values of `A`, `B`, and
     * `D`, along with the maximum value of `C`, then the inequality
     * holds for all values.
     *
     * This extension requires little to no performance overhead, and
     * may be enabled by default in future releases.
     */
    kComparisonOfProductAndSum = (1 << 3),
  };

  /*! \brief Enable an optional extension or extensions
   *
   * \param flags A bitwise OR of all optional extensions that should
   * be enabled.
   */
  TVM_DLL void SetEnabledExtensions(Extension flags);

  /*! \brief Return the currently enabled extensions */
  TVM_DLL Extension GetEnabledExtensions() const;

  /*! \brief Return the statistics counters */
  TVM_DLL ffi::ObjectRef GetStatsCounters() const;

  /*! \brief Reset the statistics counters */
  TVM_DLL void ResetStatsCounters();

  /*! \brief Set the maximum allowed number of rewrite steps
   *
   * By default, the simplifier may perform as many steps as are
   * required.  If a positive limit is set, then the simplifier will
   * throw an exception when exceeding that number of rewrite steps.
   * This allows tests to guard against performance regressions.
   *
   * Note: To maintain accurate usage counters, `Analyzer` instances
   * should be re-used wherever possible.  For example, TIR
   * transformations should declare a single `Analyzer` that is used
   * throughout the pass.  Internal helper functions that only borrow
   * the analyzer temporarily may receive the underlying `AnalyzerObj*`
   * from their calling scope.
   */
  TVM_DLL void SetMaximumRewriteSteps(int64_t maximum);

 private:
  friend class AnalyzerObj;
  friend class ConstraintContext;
  friend class CanonicalSimplifier;
  explicit RewriteSimplifier(AnalyzerObj* parent);
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
   * \param allow_override whether we allow override of existing information.
   */
  TVM_DLL void Update(const Var& var, const PrimExpr& new_expr, bool allow_override = false);

 private:
  friend class AnalyzerObj;
  friend class ConstraintContext;
  explicit CanonicalSimplifier(AnalyzerObj* parent);
  TVM_DLL ~CanonicalSimplifier();
  class Impl;
  /*! \brief Internal impl */
  Impl* impl_;
};

/*! \brief Structure for representing result of known
 *
 * Values are assigned to allow these flags to be used in bitwise
 * operations.
 */
enum class CompareResult : int {
  kInconsistent = 0,
  kEQ = 1,
  kLT = 2,
  kLE = 3,
  kGT = 4,
  kGE = 5,
  kNE = 6,
  kUnknown = 7
};

inline constexpr CompareResult operator&(CompareResult lhs, CompareResult rhs) {
  return CompareResult(static_cast<int>(lhs) & static_cast<int>(rhs));
}
inline constexpr CompareResult operator|(CompareResult lhs, CompareResult rhs) {
  return CompareResult(static_cast<int>(lhs) | static_cast<int>(rhs));
}

/*!
 * \brief Using previously specified knowns, compare the expressions provided
 *
 * Given known expressions [(a OP b), (b OP c), ..., (y OP z)], search
 * for a known result for `(a OP z)`.
 */
class TransitiveComparisonAnalyzer {
 public:
  /* \brief Using previously specified knowns, compare the expressions provided
   *
   * \param lhs The left-hand side of the comparison
   *
   * \param rhs The right-hand side of the comparison
   *
   * \param propagate_inequalities If true, attempt to find a sequence
   * of transitive inequalities that allow the lhs and rhs to be
   * compared.  If false, only use the known comparison that have been
   * directly provided.  Using `propagate_inequalities = false` is
   * roughly equivalent to comparing against all known inequality
   * expressions using `ExprDeepEqual`, but also allows for constant
   * offsets on either side of the inequality.
   *
   * \return The most specific result that can be proven about the
   * comparison.  If nothing can be proven, returns kUnknown.
   */
  TVM_DLL CompareResult TryCompare(const PrimExpr& lhs, const PrimExpr& rhs,
                                   bool propagate_inequalities = true);

  /*! \brief Bind a variable as being equal to a known expression
   *
   * \param var The variable of interest.
   * \param expr The bound expression
   * \param allow_override Whether to allow override of existing information.
   */
  TVM_DLL void Bind(const Var& var, const PrimExpr& expr, bool allow_override = false);

  /*! \brief Bind a variable as being within a specified range
   *
   * \param var The variable of interest.
   * \param range The known range
   * \param allow_override Whether to allow override of existing information.
   */
  TVM_DLL void Bind(const Var& var, const Range& range, bool allow_override = false);

  /*!
   * \brief Update the internal state to enter constraint.
   * \param constraint A constraint expression.
   *
   * \return an exit function that must be called to cleanup the constraint can be nullptr.
   */
  TVM_DLL std::function<void()> EnterConstraint(const PrimExpr& constraint);

 private:
  friend class AnalyzerObj;
  friend class ConstraintContext;
  TransitiveComparisonAnalyzer();
  TVM_DLL ~TransitiveComparisonAnalyzer();
  class Impl;
  /*! \brief Internal impl */
  std::unique_ptr<Impl> impl_;
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
  TVM_DLL IntSet operator()(const PrimExpr& expr, const ffi::Map<Var, IntSet>& dom_map);

  /*!
   * \brief Find a symbolic integer set that contains all possible
   *        values of expr given the domain of each variables, using
   *        the domain map defined by bound variables.
   *
   * \param expr The expression of interest.
   * \return the result of the analysis.
   */
  TVM_DLL IntSet operator()(const PrimExpr& expr);

  /*!
   * \brief Update binding of var to a new expression.
   *
   * \param var The variable of interest.
   * \param new_interval_set The set of allowed values for this var.
   * \param allow_override whether we allow override of existing information.
   */
  TVM_DLL void Update(const Var& var, const IntSet& new_interval_set, bool allow_override = false);

  /*!
   * \brief Update binding of var to a new expression.
   *
   * \param var The variable of interest.
   * \param new_range The range of allowed values for this var.
   * \param allow_override whether we allow override of existing information.
   */
  TVM_DLL void Bind(const Var& var, const Range& new_range, bool allow_override = false);

  std::function<void()> EnterConstraint(const PrimExpr& constraint);

 private:
  friend class AnalyzerObj;
  explicit IntSetAnalyzer(AnalyzerObj* parent);
  TVM_DLL ~IntSetAnalyzer();
  class Impl;
  /*! \brief Internal impl */
  Impl* impl_;
};

class Z3Prover {
 public:
  /*!
   * \brief Update binding of var to a new expression.
   *
   * \param var The variable of interest.
   * \param new_range The range of allowed values for this var.
   * \param allow_override whether we allow override of existing information.
   */
  TVM_DLL void Bind(const Var& var, const Range& new_range, bool allow_override = false);

  /*!
   * \brief Update binding of var to a new expression.
   *
   * \param var The variable of interest.
   * \param expr The bound expression.
   * \param allow_override whether we allow override of existing information.
   */
  TVM_DLL void Bind(const Var& var, const PrimExpr& expr, bool allow_override = false);

  /*!
   * \brief Whether the Z3 backend is compiled into this build (USE_Z3=ON).
   *
   * \return true if the real Z3 prover is available, false for the stub.
   */
  TVM_DLL bool IsEnabled() const;

  /*!
   * \brief Whether can we prove expr is always true.
   *
   * \param expr The expression.
   * \return Whether we can prove it.
   */
  TVM_DLL bool CanProve(const PrimExpr& expr);

  /*!
   * \brief Update the internal state to enter constraint.
   *
   * \param constraint A constraint expression.
   * \return an exit function that must be called to cleanup the constraint can be nullptr.
   */
  std::function<void()> EnterConstraint(const PrimExpr& constraint);

  /*!
   * \brief Get the SMTLIB2 representation of the current context.
   *
   * \param expr The optional expression to check.
   * \return The SMTLIB2 string.
   */
  ffi::String GetSMTLIB2(const ffi::Optional<PrimExpr> expr);

  /*!
   * \brief Get statistics about Z3 prover.
   *
   * \return The statistics string.
   */
  ffi::String GetStats();

  /*!
   * \brief Set timeout in milliseconds for Z3 prover.
   *
   * \param timeout_ms The timeout in milliseconds.
   */
  void SetTimeoutMs(unsigned timeout_ms);

  /*!
   * \brief Set resource limitation for Z3 prover.
   *
   * \param rlimit the resource limitation.
   */
  void SetRLimit(unsigned rlimit);

  /*!
   * \brief Get the Z3 model for the given expression if satisfiable.
   *
   * \param expr The expression to get the model for.
   * \return The model as a string.
   */
  ffi::String GetModel(const PrimExpr& expr);

  /*!
   * \brief Count the number of integer values that satisfy the current constraints.
   *
   * This method uses Z3's model enumeration to count how many distinct values of
   * the given variable satisfy all current constraints.
   *
   * \param var The variable to count satisfying values for.
   * \param max_count Maximum number of solutions to enumerate.
   * \param min_consecutive Minimum consecutive count requirement.
   * \return The number of distinct values that satisfy the constraints, or a negative error code.
   */
  TVM_DLL int64_t CountSatisfyingValues(const Var& var, int64_t max_count = 2048,
                                        int64_t min_consecutive = 1);

 private:
  friend class Analyzer;
  explicit Z3Prover(AnalyzerObj* parent);
  TVM_DLL ~Z3Prover();
  void CopyFrom(const Z3Prover& other);
  class Impl;
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
class TVM_DLL AnalyzerObj : public ffi::Object {
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
  /*! \brief sub-analyzer transitive comparisons */
  TransitiveComparisonAnalyzer transitive_comparisons;
  /*! \brief sub-analyzer using Z3 */
  Z3Prover z3_prover;
  /*! \brief constructor */
  AnalyzerObj();
  /*!
   * \brief Mark the value as non-negative value globally in analyzer.
   *
   * Only call this function if the non-neg condition is global and
   * not context-dependent.
   *
   * This function does best-effort propagations to the sub-analyzers
   *
   * \note We expose this function because non-negative global values,
   * such as symbolic buffer shapes in function arguments are really
   * important to ensure the best simplification, and usually they
   * can be handled in a simpler way than the generic constraints.
   *
   * This function may call into the Update function of the sub-analyzers.
   */
  void MarkGlobalNonNegValue(const PrimExpr& value);
  /*!
   * \brief Notify all the sub-analyzers that var
   *        is created and binded to expr.
   *
   *  Each var can only be bound once.
   *
   * \param var The variable.
   * \param expr The expression we bind to.
   * \param allow_override Whether we allow overriding an existing var's
   *        expression. This option should not be used if there is any dependency
   *        between variables.
   */
  void Bind(const Var& var, const PrimExpr& expr, bool allow_override = false);
  /*!
   * \brief Notify all the sub-analyzers that var
   *        is created and bound to a range.
   *
   *  Each var can only be bound once.
   *
   * \param var The variable.
   * \param range The range we bind to.
   * \param allow_override Whether we allow overriding an existing var's
   *        expression. This option should not be used if there is any dependency
   *        between variables.
   */
  void Bind(const Var& var, const Range& range, bool allow_override = false);
  /*!
   * \brief Bind all the vars in the Map
   *
   * \param variables The {variable -> range} map.
   * \param allow_override Whether we allow overriding an existing var's
   *        expression. This option should not be used if there is any dependency
   *        between variables.
   */
  void Bind(const ffi::Map<Var, Range>& variables, bool allow_override = false);
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
   * \brief Whether can we prove lhs == rhs.
   *
   * \param lhs The input lhs.
   * \param rhs The input rhs.
   * \return Whether we can prove lhs == rhs.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  bool CanProveEqual(const PrimExpr& lhs, const PrimExpr& rhs);
  /*!
   * \brief Whether we can prove lhs is smaller than possibly symbolic shape.
   *
   * By calling this function, the caller gives an extra hint that shape > 0,
   * because it appeared in buffer shape.
   *
   * This is useful to prove condition such as 32 <= 32 * n where the 32 * n
   * is known to be a shape. Use this routine to reduce the symbolic comparisons
   * in buffer compaction.
   *
   * The underlying analyzer will use the kSymbolicBound proof.
   *
   * \param lhs The input lhs.
   * \param shape The symbolic shape.
   * \return Whether we can prove lhs <= shape.
   */
  bool CanProveLessEqualThanSymbolicShapeValue(const PrimExpr& lhs, const PrimExpr& shape);
  /*!
   * \brief Whether can we prove condition.
   *
   * \param cond The expression to be proved.
   * \param strength the strength of the prove.
   *
   * \return The result.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   * Do not use strength beyond default in sub-analyzers and
   * only use it in top-level predicate analysis.
   */
  bool CanProve(const PrimExpr& cond, ProofStrength strength = ProofStrength::kDefault);

  /*!
   * \brief Simplify expr.
   *
   * \param expr The expression to be simplified.
   * \param steps The simplification runs in the order of
   *        rewrite_simplify (step 1) -> canonical_simplify (step 2) ->
   *        rewrite_simplify (step 3) -> canonical_simplify (step 4) -> ...
   *        param steps controls how many steps to run.
   *        Default is 2, i.e., rewrite_simplify + canonical_simplify.
   * \return The result.
   *
   * \note Analyzer will call into sub-analyzers to get the result.
   */
  PrimExpr Simplify(const PrimExpr& expr, int steps = 2);

  /*!
   * \brief Analyzer methods update facts, constraints, caches, and stats.
   *
   * Marking the object mutable makes the `Analyzer` ObjectRef expose a
   * non-const `operator->`, so APIs can take `const Analyzer&` while still
   * allowing calls such as `analyzer->Bind(...)`.
   * `const Analyzer&` keeps the handle itself from being rebound; it does
   * not make the underlying AnalyzerObj immutable.
   */
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("arith.Analyzer", AnalyzerObj, ffi::Object);
};

/*!
 * \brief Managed reference to AnalyzerObj.
 *
 * Analyzer is a lightweight, reference-counted handle around a heap-allocated
 * AnalyzerObj. Because it is now a first-class FFI object, an Analyzer can be
 * passed across the tvm-ffi boundary (e.g. handed from Python into a C++ pass)
 * and shared, so that accumulated bindings/constraints persist across calls.
 * Copying an Analyzer copies the handle, and both handles share the same
 * mutable AnalyzerObj state.
 * This is not a deep copy of analyzer facts or caches.
 *
 * \sa AnalyzerObj
 */
class Analyzer : public ffi::ObjectRef {
 public:
  /*! \brief Default-construct a fresh analyzer (allocates an AnalyzerObj). */
  Analyzer() : Analyzer(ffi::make_object<AnalyzerObj>()) {}
  explicit Analyzer(ffi::ObjectPtr<AnalyzerObj> n) : ffi::ObjectRef(std::move(n)) {
    TVM_FFI_ICHECK(this->get() != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Analyzer, ffi::ObjectRef, AnalyzerObj);
};

/*!
 * \brief Constraint context.
 *
 * \code
 *
 *  Var x("x");
 *  arith::Analyzer analyzer;
 *  {
 *    With<arith::ConstraintContext> scope(analyzer, tvm::floormod(x, 3) == 0);
 *    TVM_FFI_ICHECK_EQ(analyzer->modular_set(x)->coeff, 3);
 *  }
 *  // constraint no longer in effect.
 *  TVM_FFI_ICHECK_NE(analyzer->modular_set(x)->coeff, 3);
 *
 * \endcode
 */
class ConstraintContext {
 private:
  // declare friend to enable with.
  friend class With<ConstraintContext>;
  /*!
   * \brief Construct a constraint context.
   * \param analyzer The analyzer whose context is updated. The context
   *        keeps a reference to the analyzer while the scope is active.
   * \param constraint The constraint to be applied.
   */
  ConstraintContext(const Analyzer& analyzer, PrimExpr constraint)
      : ConstraintContext(analyzer, std::move(constraint), false) {}
  /*!
   * \brief Construct a constraint context.
   * \param analyzer The analyzer whose context is updated. The context
   *        keeps a reference to the analyzer while the scope is active.
   * \param constraint The constraint to be applied.
   * \param is_assume Whether the constraint comes from an assumption.
   */
  ConstraintContext(const Analyzer& analyzer, PrimExpr constraint, bool is_assume)
      : analyzer_(analyzer), constraint_(std::move(constraint)), is_assume_(is_assume) {}
  /*!
   * \brief Construct a constraint context from a borrowed analyzer object.
   * \param analyzer The borrowed analyzer object.
   * \param constraint The constraint to be applied.
   *
   * This overload is for internal callers that already operate on AnalyzerObj*.
   */
  ConstraintContext(AnalyzerObj* analyzer, PrimExpr constraint)
      : ConstraintContext(ffi::GetRef<Analyzer>(analyzer), std::move(constraint), false) {}
  /*!
   * \brief Construct a constraint context from a borrowed analyzer object.
   * \param analyzer The borrowed analyzer object.
   * \param constraint The constraint to be applied.
   * \param is_assume Whether the constraint comes from an assumption.
   */
  ConstraintContext(AnalyzerObj* analyzer, PrimExpr constraint, bool is_assume)
      : ConstraintContext(ffi::GetRef<Analyzer>(analyzer), std::move(constraint), is_assume) {}
  // enter the scope.
  void EnterWithScope();
  // exit the scope.
  void ExitWithScope();
  /*! \brief Analyzer kept alive while the context is active. */
  Analyzer analyzer_;
  /*! \brief The constraint */
  PrimExpr constraint_;
  /*! \brief functions to be called in recovery */
  std::vector<std::function<void()>> recovery_functions_;
  /*! \brief Whether the constraint comes from an assumption. */
  bool is_assume_;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_ANALYZER_H_
