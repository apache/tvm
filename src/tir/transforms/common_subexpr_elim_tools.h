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
 * \file common_subexpr_elim_tools.h
 * \brief Interface of analysis tools and utility functions used
           by the Common Subexpression Elimination (CSE) pass.
 */

#ifndef TVM_TIR_TRANSFORMS_COMMON_SUBEXPR_ELIM_TOOLS_H_
#define TVM_TIR_TRANSFORMS_COMMON_SUBEXPR_ELIM_TOOLS_H_

#include <tvm/runtime/container/string.h>
#include <tvm/tir/analysis.h>  // For the ExprDeepEqual analysis
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>  // For the class StmtExprVisitor

#include <optional>
#include <unordered_map>  // For the hashtable datatype
#include <utility>        // For pairs datatype
#include <vector>

namespace tvm {
namespace tir {

/*!
 * \brief A computation table is a hashtable which associates to each expression being computed
          a number (which is the number of time that it is computed)
          It is important to note that the hash used is a StructuralHash (and not an ObjectPtrHash)
          as we need to hash similarly deeply equal terms.
          The comparison used is ExprDeepEqual, which is stricter than StructuralEqual (as it does
          not do variables remapping), so it is compatible with StructuralHash (intended to be used
          with StructuralEqual).
 */
using ComputationTable = std::unordered_map<PrimExpr, size_t, StructuralHash, ExprDeepEqual>;

/*!
 * \brief A cache of computations is made of a pair of two hashtables, which respectively associate
          to each statement or expression of the program its computation table. Its purpose is
          to avoid the CSE pass from recomputing repeatedly the same tables of computations.
 */
struct ComputationCache {
  // Part of the cache for statements
  // It maps each known statement to its computation table
  std::unordered_map<Stmt, ComputationTable, ObjectPtrHash, ObjectPtrEqual>
      cache_stmt_table_computations_;

  // Part of the cache for expressions
  // It maps each known expression to its computation table
  std::unordered_map<PrimExpr, ComputationTable, ObjectPtrHash, ObjectPtrEqual>
      cache_expr_table_computations_;
};

/*!
 * \brief Visitor which returns in a hashtable the (syntatic) computations done by an expression
          or by a statement.
 * \note Computations here are considered syntactically, meaning that semantically equivalent
          computations that are not syntactically the same are not merged together.
 */
class ComputationsDoneBy : public StmtExprVisitor {
 public:
  // Toplevel (static) methods
  static ComputationTable GetComputationsDoneBy(
      const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
      std::function<bool(const PrimExpr&)> can_contain_computations);
  static ComputationTable GetComputationsDoneBy(
      const Stmt& stmt, std::function<bool(const PrimExpr&)> is_eligible_computation,
      std::function<bool(const PrimExpr&)> can_contain_computations);

 protected:
  // Constructor
  ComputationsDoneBy(std::function<bool(const PrimExpr&)> is_eligible_computation,
                     std::function<bool(const PrimExpr&)> can_contain_computations);

  void VisitExpr(const PrimExpr& expr) override;
  void VisitStmt(const Stmt& stmt) override;

  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const WhileNode* op) override;

 private:
  static ComputationTable ComputationsDoneByChildrenOf(
      const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
      std::function<bool(const PrimExpr&)> can_contain_computations);
  static ComputationTable ComputationsDoneByChildrenOf(
      const Stmt& stmt, std::function<bool(const PrimExpr&)> is_eligible_computation,
      std::function<bool(const PrimExpr&)> can_contain_computations);

  // The predicate used for knowing which computations are eligible
  std::function<bool(const PrimExpr&)> is_eligible_computation_;
  // The predicate used for knowing in which nodes we can search for eligible computations
  std::function<bool(const PrimExpr&)> can_contain_computations_;
  // The object being constructed and "returned" by the VisitExpr()/VisitStmt() methods
  ComputationTable table_of_computations_;
  // Cache for preventing to compute repeatedly the computations done by the same stmt or expr
  static ComputationCache cache_;
};

/*!
 * \brief Visitor that computes the *direct* subexpressions of a given expression.
 * \note Returns only the direct subexpressions of the given expressions, not all the subexprs.
          So for instance, for (A+(B+C)) it will return A and (B+C) if they are eligible,
          but not B and C.
 */
class DirectSubexpr : public ExprVisitor {
 public:
  // Toplevel (static) function
  static std::vector<PrimExpr> GetDirectSubexpressions(
      const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
      std::function<bool(const PrimExpr&)> can_contain_computations);

 protected:
  // Constructor
  DirectSubexpr(std::function<bool(const PrimExpr&)> is_eligible_computation,
                std::function<bool(const PrimExpr&)> can_contain_computations);

  void VisitExpr(const PrimExpr& expr) override;

 private:
  // The predicate used for knowing which computations are eligible
  std::function<bool(const PrimExpr&)> is_eligible_computation_;
  // The predicate used for knowing in which nodes we can search for eligible subexpressions
  std::function<bool(const PrimExpr&)> can_contain_computations_;

  // We haven't entered the VisitExpr() method yet
  bool entered_ = false;
  // The vector of direct subexpressions that we are building
  std::vector<PrimExpr> direct_subexpr_;
};

/*!
 * \brief Visitor which tells if a given expression or statement uses a given variable name.
          This is used by the CSE pass to make sure that we do not reuse existing names,
          even though having the same name does not mean that it's the same variable, but it's
          clearer for dumps.
 */
class UsesVarName : public StmtExprVisitor {
 public:
  // Toplevel (static) methods
  static bool ExprUsesVarName(const PrimExpr& expr, String var_name);
  static bool StmtUsesVarName(const Stmt& stmt, String var_name);

 protected:
  // Constructor
  explicit UsesVarName(String var_name);

  void VisitExpr(const PrimExpr& expr) override;
  void VisitStmt(const Stmt& stmt) override;

 private:
  String var_name_;
  bool uses_var_name_ = false;
};

/*!
 * \brief Various utility functions for the CSE pass
 */
void PrintComputationTable(const ComputationTable& table);

using MaybeValue = std::optional<PrimExpr>;

bool EqualTerms(const PrimExpr& a, const PrimExpr& b);
// Used for deciding the (decidable) equivalence relation
PrimExpr NormalizeTerm(const PrimExpr& expr, bool do_normalization);
// The equivalence relation, which is the syntactical equality when `identify_equiv_terms` is false
bool EquivalentTerms(const PrimExpr& a, const PrimExpr& b, bool identify_equiv_terms);
std::vector<std::pair<PrimExpr, size_t>> SyntacticToSemanticComputations(
    const ComputationTable& table, bool identify_equiv_terms);
bool PredicateIntroVarForComputation(const PrimExpr& computation, size_t nb_times_seen);

// Polymorphic (functional) map on a vector, which builds a news vector with the same number of
// elements, where each element is the application of a given function on the corresponding element
// in the input vector.
template <typename A, typename B>
std::vector<B> VectorMap(const std::vector<A>& input, std::function<B(const A&)> fun) {
  std::vector<B> result;
  size_t size = input.size();
  // For efficiency, allocate immediately the size needed as the result will have
  // the same size as the input
  result.reserve(size);

  for (size_t i = 0; i < size; i++) {
    result.push_back(fun(input[i]));
  }

  return result;
}
// Explicitely instanciate the template function for A=std::pair<Var,MaybeValue> and B=Var
template std::vector<Var> VectorMap(const std::vector<std::pair<Var, MaybeValue>>&,
                                    std::function<Var(const std::pair<Var, MaybeValue>&)>);

void InsertElemToSortedSemanticComputations(std::vector<std::pair<PrimExpr, size_t>>* sorted_vec,
                                            const std::pair<PrimExpr, size_t>& pair);

void InsertVectorToSortedSemanticComputations(std::vector<std::pair<PrimExpr, size_t>>* sorted_vec,
                                              const std::vector<PrimExpr>& vec_to_add,
                                              bool identify_equiv_terms, size_t increase_count = 1);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORMS_COMMON_SUBEXPR_ELIM_TOOLS_H_
