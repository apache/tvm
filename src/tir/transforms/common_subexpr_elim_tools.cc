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
* \file common_subexpr_elim_tools.cc
* \brief Implementation of analysis tools and utility functions used
          by the Common Subexpression Elimination (CSE) pass.
*/

#include "common_subexpr_elim_tools.h"

#include <tvm/ir/transform.h>  // For the class Pass and the class PassContext
#include <tvm/runtime/container/string.h>
#include <tvm/tir/analysis.h>  // For the ExprDeepEqual analysis
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>  // For the class PrimFunc
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>  // For the declaration of the pass

#include <algorithm>      // For std::find_if
#include <unordered_map>  // For the hashtable datatype
#include <vector>

#include "../analysis/check_contains.h"  // For the CheckContains analysis

namespace tvm {
namespace tir {

// cache_ is a static variable of the class ComputationsDoneBy, and C++ requires to define here
// such static attribute, otherwise it causes a linking error.
CacheOfComputations ComputationsDoneBy::cache_;

/* ********************************** Class ComputationsDoneBy **********************************
*********************************************************************************************** */

/* This utility class of the CSE pass offers a way of knowing the computations done by a given
   statement or expression. A "computation" here is a syntatical entity, represented by a PrimExpr.
   This analysis returns a hashtable associating PrimExpr (a computation done) to a number (which
   is the number of time that this computation is being computed).
   This analysis is used by the CSE pass in order to find potential candidates for being introduced
   into new variables (after having merged semantically equivalent computations).

   This analysis is parametrized by two predicates : `is_eligible_computation` and
   `can_contain_computations`.
   The first one helps to select only "eligible" computations, and the second one helps to only
   select computations that are located at appropriate location (i.e., it tells in which nodes the
   analysis can recurse). The user of the class must define these notions of "eligible computation"
   and of "nodes that can contain eligibile computations" for his own use case.

   - On an statement, this analysis returns the union of all the computations that appear in its
   child nodes (ie, the union of the results of the recursive calls).
   For instance, on the input statement [let a = x+y in Mem[i1+i2] = a+b] it will return (x+y),
   (i1+i2) and (a+b) when used with typical predicates.
   - On an expression, this analysis returns the expression itself, except if it is not eligible
   for being introduced by the CSE pass into a variable according to `is_eligible_computation_`
   (often because it's a load node or a function call node for instance), in which case it will
   return the union of the recursive calls on its children, as long as the other predicate
   `can_contain_computations` evaluates to true to let the algorithm recurse deeper.
   With typical predicates, on the expression ((w+x)+(y+z)) it will return only the expression
   itself. But on the expression Load[i1+i2] it might return only (i1+i2) as the full Load node
   might not be eligible.

   This class uses an internal cache of results, so that if one queries it several times on the
   same statement or expression, it will just retrieve the result from its internal cache.
   That avoids some systematic recomputations, which would otherwise happen as the CSE pass first
   analyses the program at the toplovel (asking for the computations done by the root), and then
   dives deeper and deeper into the program, asking for the computations done by the children of
   the root, which were necessarly previously obtained when computing the computations done by the
   root (as the computations done by the root are by definition the union of the computations done
   by the children nodes).

   The somehow difficult aspect of the implementation is the interaction between this caching of
   results, and the fact that the VisitStmt()/VisitExpr() of an analyzer (a StmtExprVisitor) are
   void methods which can't return anything, and instead need to accumulate a result into a member
   variable, which is called `table_of_computations_` here.

   In particular, as the specialized methods (all the VisitStmt_() and VisitExpr_() methods), just
   call VisitStmt()/VisitExpr() on all the children nodes within the same instance, if we don't
   want to override each of these specialized methods to change this behaviour, then
   `table_of_computations_` will necessary be shared by all the children of a given nodes.
   That requires to be careful when trying to write into the cache.
*/

/*!
 * \brief Does the union of two table of computations.
 * \param tableMain One of the two tables. The union will be written into it.
 * \param tableAux The other table, which won't change.
 */
void UnionOfTablesOfComputations(TableOfComputations& table_main,
                                 const TableOfComputations& table_aux) {
  // Adds each element of the second table to the first one
  for (const auto& current : table_aux) {
    table_main[current.first] += current.second;
  }
}

/*!
 * \brief Toplevel (static) method for a PrimExpr
 * \param expr The expr for which we want to know the computations done
 * \param is_eligible_computation The predicate which decides if an expression is eligible for
                                  being introduced in a new variable
 * \param can_contain_computations The predicate which decides if an expression can contain an
                                    eligible computation
 */
TableOfComputations ComputationsDoneBy::GetComputationsDoneBy(
    const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  // Chunk for avoiding the lookup (and writing) in the cache for an atom (constant or variable),
  // for which the table of computations is empty.
  // (We don't want to use a "line of cache" of that, as that would cost an empty table of
  // computations in memory for absolutely no gain)
  if (expr.as<IntImmNode>() != nullptr || expr.as<FloatImmNode>() != nullptr ||
      expr.as<StringImmNode>() != nullptr || expr.as<VarNode>() != nullptr) {
    // Return an empty table
    return {};
  }

  // See if we have already computed the (table of) computations done by `expr`
  auto it_table_expr = cache_.cache_expr_table_computations_.find(expr);
  if (it_table_expr != cache_.cache_expr_table_computations_.end()) {
    // then we just return it
    return it_table_expr->second;
  }

  // Otherwise we will need to compute it, by using an instance of the class ComputationsDoneBy
  // (as we are currently in a static method)
  ComputationsDoneBy computations_done_by(is_eligible_computation, can_contain_computations);
  // Call the VisitExpr() method on it to start the visit
  computations_done_by.VisitExpr(expr);
  // Copy the `table_of_computations_` (that `computations_done_by` has computed) into the cache
  // for the future queries
  cache_.cache_expr_table_computations_[expr] = computations_done_by.table_of_computations_;

  return computations_done_by.table_of_computations_;
}

/*!
 * \brief Toplevel (static) method for a Stmt
 * \param stmt The stmt for which we want to know the computations done
 * \param is_eligible_computation The predicate which decides if an expression is eligible for
                                  being introduced in a new variable
 *  \param can_contain_computations The predicate which decides if an expression can contain an
                                    eligible computation
 */
TableOfComputations ComputationsDoneBy::GetComputationsDoneBy(
    const Stmt& stmt, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  // See if we have already computed the (table of) computations done by `stmt`
  auto it_table_stmt = cache_.cache_stmt_table_computations_.find(stmt);
  if (it_table_stmt != cache_.cache_stmt_table_computations_.end()) {
    // then we just return it
    return it_table_stmt->second;
  }

  // Otherwise we will need to compute it, by using an instance of the class ComputationsDoneBy
  // (as we are currently in a static method)
  ComputationsDoneBy computations_done_by(is_eligible_computation, can_contain_computations);
  // Call the VisitStmt() method on it to start the visit
  computations_done_by.VisitStmt(stmt);
  // Copy the `table_of_computations_` that `computations_done_by` has computed into the cache
  // for the future queries
  cache_.cache_stmt_table_computations_[stmt] = computations_done_by.table_of_computations_;

  return computations_done_by.table_of_computations_;
}

/*!
 * \brief Protected constructor of ComputationsDoneBy.
 * \param is_eligible_computation The predicate which decides if an expression is eligible for
                                  being introduced in a new variable
 * \param can_contain_computations The predicate which decides if an expression can contain an
                                    eligible computation
 */
ComputationsDoneBy::ComputationsDoneBy(
    std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations)
    : is_eligible_computation_(is_eligible_computation),
      can_contain_computations_(can_contain_computations) {}

/*!
 * \brief The method which overrides the generic dispatcher of StmtExprVisitor for expressions
 * \param expr The expression to visit
 */
void ComputationsDoneBy::VisitExpr(const PrimExpr& expr) {
  // Chunk for avoiding the lookup (and writing) in the cache for an atom (constant or variable),
  // for which the table of computations is empty.
  // (We don't want to use a "line of cache" of that, as that would cost an empty table of
  // computations in memory for absolutely no gain)
  if (expr.as<IntImmNode>() != nullptr || expr.as<FloatImmNode>() != nullptr ||
      expr.as<StringImmNode>() != nullptr || expr.as<VarNode>() != nullptr) {
    return;
  }

  // See if we have already computed the (table of) computations done by `expr`
  auto it_table_expr = cache_.cache_expr_table_computations_.find(expr);
  if (it_table_expr != cache_.cache_expr_table_computations_.end()) {
    // We need to do the union with `table_of_computations_` instead of just writing into it,
    // because some other childs might have added things into it too. The reason for that is
    // that `table_of_computations_` is shared between the child nodes of a given expression.
    UnionOfTablesOfComputations(table_of_computations_, it_table_expr->second);
    return;
  }

  // If we reach this point, it means that we have never computed before the computations done
  // by 'expr' and will do so now.

  // If the given expression is an eligible computation, we simply "return it" by adding it into
  // the "result variable" that `table_of_computations_` is.
  if (is_eligible_computation_(expr)) {
    // We can add `expr` to the table of computations
    table_of_computations_[expr]++;
    return;
  }

  // If we reach this point, then the given expression is not an eligible computation.
  // But perhaps we have the right to dive into it to find some smaller eligible computations
  if (can_contain_computations_(expr)) {
    TableOfComputations temp =
        ComputationsDoneByChildrenOf(expr, is_eligible_computation_, can_contain_computations_);
    // We need to do the union with `table_of_computations_` instead of just writing into it,
    // because some other childs might have added things into it too. The reason for that is
    // that `table_of_computations_` is shared between the child nodes of a given expression.
    UnionOfTablesOfComputations(table_of_computations_, temp);
    return;
  }

  // Note that we do not continue by calling the general disptacher
  // StmtExprVisitor::VisitExpr(expr) as we want the full computations, not their subexpressions.
}

/*!
 * \brief The method which overrides the generic dispatcher of StmtExprVisitor for statements
 * \param stmt The statement to visit
 */
void ComputationsDoneBy::VisitStmt(const Stmt& stmt) {
  // See if we have already computed the (table of) computations done by `stmt`
  auto it_table_stmt = cache_.cache_stmt_table_computations_.find(stmt);
  if (it_table_stmt != cache_.cache_stmt_table_computations_.end()) {
    // We need to do the union with `table_of_computations_` instead of just writing into it,
    // because some other childs might have added things into it too. The reason for that is
    // that `table_of_computations_` is shared between the child nodes of a given statement.
    UnionOfTablesOfComputations(table_of_computations_, it_table_stmt->second);
    return;
  }

  // If we reach this point, it means that we have never computed before the computations done
  // by `stmt` and will do so now.

  // The computations done by a Stmt node are just the ones done by its children
  TableOfComputations temp =
      ComputationsDoneByChildrenOf(stmt, is_eligible_computation_, can_contain_computations_);
  // We need to do the union with `table_of_computations_` instead of just writing into it,
  // because some other childs might have added things into it too. The reason for that is
  // that `table_of_computations_` is shared between the child nodes of a given expression.
  UnionOfTablesOfComputations(table_of_computations_, temp);
}

/*!
 * \brief Static method that returns the computations done by the children of an expression.
 * \param expr The expression to analyze
 * \param is_eligible_computation The predicate which decides if an expression is eligible for
      being introduced in a new variable
 * \param can_contain_computations The predicate which decides if an expression can contain an
      eligible computation
 * \return The hashtable containing the (syntactic) computations done by children nodes of `expr`
 */
TableOfComputations ComputationsDoneBy::ComputationsDoneByChildrenOf(
    const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  // We will be using an instance of the class ComputationsDoneBy for the child nodes
  // (ie, they will share the "result" that `table_of_computations_` is)
  ComputationsDoneBy computations_done_by(is_eligible_computation, can_contain_computations);
  // Calls the *dispatcher* (not the overriden method)
  computations_done_by.StmtExprVisitor::VisitExpr(expr);
  // Now we can copy `table_of_computations_` into the cache for the future queries
  // Note : in the table, the computations done by `expr` is set to the computations done by its
  // children, because otherwise we would not have needed to compute them.
  cache_.cache_expr_table_computations_[expr] = computations_done_by.table_of_computations_;

  return computations_done_by.table_of_computations_;
}

/*!
 * \brief Static method that returns the computations done by the children of a statement.
 * \param stmt The statement to analyze.
 * \param is_eligible_computation The predicate which decides if an expression is eligible for
                                  being introduced in a new variable
 * \param can_contain_computations The predicate which decides if an expression can contain an
                                    eligible computation
 * \return The hashtable contaning the (syntactic) computations done by children nodes of `stmt`
 */
TableOfComputations ComputationsDoneBy::ComputationsDoneByChildrenOf(
    const Stmt& stmt, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  // We will be using an instance of the class ComputationsDoneBy for the child nodes
  // (ie, they will share the "result" that `table_of_computations_` is)
  ComputationsDoneBy computations_done_by(is_eligible_computation, can_contain_computations);
  // Calls the *dispatcher* (not the overriden method)
  computations_done_by.StmtExprVisitor::VisitStmt(stmt);
  // So now we can copy table_of_computations_ into the cache for the future queries
  // Note : in the table, the computations done by `stmt` is set the the computations done by its
  // children, because that's exactly what we mean by "the computations of a statement".
  cache_.cache_stmt_table_computations_[stmt] = computations_done_by.table_of_computations_;

  return computations_done_by.table_of_computations_;
}

/* *********************************** Class DirectSubexpr **************************************
*********************************************************************************************** */

/* This utility class of the CSE pass offers a way of obtaining the direct subexpression
   of a given expression.
   For instance, for (A+(B+C)) it will return A and (B+C) if they are eligible, but not B and C.
   If one of the direct subexpression is not eligible, it will consider the direct subexprs of this
   uneligible expression (and etcetera if one of them is not eligible).
   But before continuing recursively on an ineligible term, it makes sure that is has the right to
   do so by checking if `can_contain_computations` evaluates to true.

   This is used by the CSE pass, which will first attempt to introduce large computations into new
   variables, and only when that's not possible (either because the computation uses some variables
   not yet within scope, or because it is not computed enough for being a good candidate), it will
   consider its direct subexpression. That avoids to compute all the subexpression at once, and
   instead evaluates them lazily, if and when needed.
*/

/*!
 * \brief Toplevel (static) function that returns the direct subexpressions of a given expression
 * \param expr The expression to analyze.
 * \param is_eligible_computation The predicate which decides if an expression is eligible for
                                  being introduced in a new variable
 * \param can_contain_computations The predicate which decides if an expression can contain an
                                    eligible computation
 * \return A vector of PrimExpr containing the direct subexpressions of `expr`
 */
std::vector<PrimExpr> DirectSubexpr::GetDirectSubexpressions(
    const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  DirectSubexpr direct_subexpr(is_eligible_computation, can_contain_computations);
  direct_subexpr.VisitExpr(expr);

  return direct_subexpr.direct_subexpr_;
}

/*!
 * \brief Protected constructor of DirectSubexpr.
 */
DirectSubexpr::DirectSubexpr(std::function<bool(const PrimExpr&)> is_eligible_computation,
                             std::function<bool(const PrimExpr&)> can_contain_computations)
    : is_eligible_computation_(is_eligible_computation),
      can_contain_computations_(can_contain_computations) {}

/*!
 * \brief The method which overrides the generic dispatcher of ExprVisitor
 * \param expr The expression to visit
 */
void DirectSubexpr::VisitExpr(const PrimExpr& expr) {
  // If we have already entered (meaning that we are not dealing with the original expression)
  if (entered_) {
    if (is_eligible_computation_(expr)) {
      direct_subexpr_.push_back(expr);
      return;
    } else {
      if (can_contain_computations_(expr)) {
        ExprVisitor::VisitExpr(expr);
      }
      return;
    }
  }

  // If we reach this point, it means that we haven't visited any child node yet, and will need
  // to dive into the expression, if it is allowed to contain eligible computations
  if (can_contain_computations_(expr)) {
    // Take note that now we have already visited some node
    entered_ = true;
    ExprVisitor::VisitExpr(expr);
  }
}

/* ************************************ Class UsesVarName *************************************
*********************************************************************************************** */

/*!
 * \brief Toplevel (static) function that tells if a given expression uses a given variable name.
 * \param expr The expression to analyze
 * \param var_name The variable name to check for
 * \return A boolean telling if `expr` uses `var_name`
 */
bool UsesVarName::ExprUsesVarName(const PrimExpr& expr, String var_name) {
  UsesVarName uses_var_name(var_name);
  uses_var_name.VisitExpr(expr);

  return uses_var_name.uses_var_name_;
}

/*!
 * \brief Toplevel (static) function that tells if a given statement uses a given variable name.
 * \param stmt The statement to analyze
 * \param var_name The variable name to check for
 * \return A boolean telling if `stmt` uses `var_name`
 */
bool UsesVarName::StmtUsesVarName(const Stmt& stmt, String var_name) {
  UsesVarName uses_var_name(var_name);
  uses_var_name.VisitStmt(stmt);

  return uses_var_name.uses_var_name_;
}

/*!
 * \brief Protected constructor of UsesVarName.
 * \param var_name The String that we are looking for
 */
UsesVarName::UsesVarName(String var_name) : var_name_(var_name) {}

/*!
 * \brief The method which overrides the generic dispatcher of StmtExprVisitor for expressions.
 * \param expr The expression to visit
 */
void UsesVarName::VisitExpr(const PrimExpr& expr) {
  if (auto var_node = expr.as<VarNode>()) {
    if (var_node->name_hint == var_name_) {
      uses_var_name_ = true;
      return;
    }
  }
  StmtExprVisitor::VisitExpr(expr);
}

/*!
 * \brief The method which overrides the generic dispatcher of StmtExprVisitor for statements.
 * \param stmt The statement to visit
 */
void UsesVarName::VisitStmt(const Stmt& stmt) {
  // We keep exploring only if `uses_var_name_` is false
  if (!uses_var_name_) {
    // and in order to do that we call the general dispatcher
    StmtExprVisitor::VisitStmt(stmt);
  }
  // As otherwise we already have our answer
}

/* ********************************** Utility functions for CSE *********************************
*********************************************************************************************** */

/*!
 * \brief Decides if two terms are equal syntactically
 */
bool EqualTerms(const PrimExpr& a, const PrimExpr& b) {
  ExprDeepEqual deep_equal_;
  return deep_equal_(a, b);
}

/*!
 * \brief Decides if two terms are equivalent semantically
 */
bool EquivalentTerms(const PrimExpr& a, const PrimExpr& b) {
  // For now, we just check the syntactic equality, but that could later become a semantic test,
  // for instance identifying computations modulo commutativity (like x+y and y+x), or modulo
  // associativity (like (x+y)+z and x+(y+z)), etc.
  return EqualTerms(a, b);
}

/*!
 * \brief Transforms a hashtable of syntactic computations into a vector or pairs
          (expression, counter) where equivalent computations are merged and their counters added.
          This function simply looks for semantically equivalent terms in order to get the real
          total number of times a computation (and semantically equivalent ones) is seen.
 * \param table The table to transform
   \note This function is needed because the advantage of the hashtable was the constant lookup.
          But in order to have this constant lookup, we could not collapse semantically equivalent
          computations.
 */
std::vector<std::pair<PrimExpr, size_t>> SyntacticToSemanticComputations(
    const TableOfComputations& table) {
  std::vector<std::pair<PrimExpr, size_t>> result;
  // table.size() is an upper-bound of the number of elements in the resulting vector,
  // as we might merge semantically equivalent computations.
  // We do this reservation even if it might reserve slightly more space than is needed in the end
  result.reserve(table.size());

  // For each element in the hashtable
  for (auto elem : table) {
    // We try to see if a semantically equivalent term is already in the resulting vector
    auto it_found = std::find_if(result.begin(), result.end(),
                                 [elem](std::pair<PrimExpr, size_t> already_seen) {
                                   return EquivalentTerms(already_seen.first, elem.first);
                                 });
    // And if so, we increase (by `elem.second`) its count
    if (it_found != result.end()) {
      it_found->second += elem.second;
    } else {
      // If we could not find a semantically equivalent term in the resulting vector, we add it
      result.push_back(elem);
    }
  }

  return result;
}

/*!
 * \brief Predicate that decides if a computation, that is seen `nb_times_seen`, should be
 introduced in a variable or not.
 */
bool PredicateIntroVarForComputation(const PrimExpr& computation, size_t nb_times_seen) {
  // This predicate could later implement something more fine grained that would take in account
  // the size of the expression too, as for instance a very large computation could be introduced
  // as soon as two occurences are seen, but a smaller one could need three or more occurences
  // for being introduced in a variable.

  // But for now, we factorize any eligible item that we see at least twice, regardless of its size
  return nb_times_seen >= 2;
}

/*!
 * \brief Inserts a pair (expr,nb) to a sorted vector of such pairs (which is sorted by decreasing
          size of expressions) and maintain the vector sorted while doing so.
 */
void InsertElemToSortedSemanticComputations(std::vector<std::pair<PrimExpr, size_t>>& sorted_vec,
                                            const std::pair<PrimExpr, size_t>& pair) {
  // Find the insertion point using std::lower_bound on a comparison that uses
  // CalculateExprComplexity(), which computes the "size" of an expr.
  // std::lower_boud returns an iterator pointing to the first element on which the comparison
  // does not return true with the given value (`pair` here), i.e, an iterator pointing to the
  // first element that is not greater or equal than `pair`, i.e, the first element that is
  // strictly smaller than `pair`.
  auto insertion_point = std::lower_bound(
      sorted_vec.begin(), sorted_vec.end(), pair,
      [](const std::pair<PrimExpr, size_t>& left, const std::pair<PrimExpr, size_t>& right) {
        return (CalculateExprComplexity(left.first) >= CalculateExprComplexity(right.first));
      });
  sorted_vec.insert(insertion_point, pair);
}

/*!
 * \brief Inserts a vector of expressions into a sorted vector of computations (which is sorted by
          decreasing size of the expression) and maintain the vector sorted while doing so.
 */
void InsertVectorToSortedSemanticComputations(std::vector<std::pair<PrimExpr, size_t>>& sorted_vec,
                                              const std::vector<PrimExpr>& vec_to_add) {
  for (auto elem_to_add : vec_to_add) {
    // See if the current element to add (or an equivalent one) is already present
    // in the sorted vector
    auto it_found = std::find_if(sorted_vec.begin(), sorted_vec.end(),
                                 [elem_to_add](std::pair<PrimExpr, size_t> elem) {
                                   return EquivalentTerms(elem.first, elem_to_add);
                                 });

    // If we found `elem_to_add` (or an equivalent expression) already in sorted_vec
    if (it_found != sorted_vec.end()) {
      // then we just increase its associated count
      it_found->second++;
    } else {
      // Otherwise we add the pair (`elem_to_add`,1) at the right place
      InsertElemToSortedSemanticComputations(sorted_vec, {elem_to_add, 1});
    }
  }
}

}  // namespace tir
}  // namespace tvm