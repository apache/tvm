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

#include <tvm/arith/analyzer.h>  // For the arith::Analyzer::Simplify() method simplifying terms
#include <tvm/ir/transform.h>    // For the class Pass and the class PassContext
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
#include <utility>
#include <vector>

#include "../analysis/check_contains.h"  // For the CheckContains analysis

namespace tvm {
namespace tir {

// cache_ is a static variable of the class ComputationsDoneBy, and C++ requires to define here
// such static attribute, otherwise it causes a linking error.
ComputationCache ComputationsDoneBy::cache_;

/* ********************************** Class ComputationsDoneBy **********************************
*********************************************************************************************** */

/* This utility class of the CSE pass offers a way of knowing the eligible computations done by a
   statement or expression. A "computation" here is a syntatical entity, represented by a PrimExpr.
   This analysis returns a hashtable associating PrimExpr (a computation done) to a number (which
   is the number of time that this computation is being seen).
   This analysis is used by the CSE pass in order to find potential candidates for being introduced
   into new variables (after having merged semantically equivalent computations).

   This analysis is parametrized by two predicates : `is_eligible_computation` and
   `can_contain_computations`.
   The first one helps to select only "eligible" computations, and the second one helps to only
   select computations that are located at appropriate location (i.e., it tells in which nodes the
   analysis can recurse). The user of the class must define these notions of "eligible computation"
   and of "nodes that can contain eligibile computations" for his own use case.

   - On an statement, this analysis often returns the union of all the computations that appear in
   its child nodes (ie, the union of the results of the recursive calls).
   For instance, on the input statement [let a = x+y in Mem[i1+i2] = a+b] it will report (x+y)
   seen once, (i1+i2) seen once, and (a+b) also seen once when used with typical predicates.
   On some nodes, it will return something more complicated that uses the intersection of the
   computations done by the children nodes.
   For instance, on the input statement [if (x+y>z) then a = x+y else a = b-x] it will return
   (x+y) seen twice but it won't report b-x as is it seen only the else branch.

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
 * \brief Does the union of two tables of computations.
 * \param table_main Pointer to one of the two tables. The union will be written into it.
 * \param table_aux The other table, which won't change.
 * \note Does it directly in the first argument A for efficiency, as the union of A and B
 *       necessarily gives something which contains A, so we avoid its copy.
 */
void UnionOfComputationTables(ComputationTable* table_main, const ComputationTable& table_aux) {
  if (table_main == nullptr) {
    return;
  }
  // Adds each element of the second table to the first one
  for (const auto& current : table_aux) {
    (*table_main)[current.first] += current.second;
  }
}

/*!
 * \brief Does the union of three tables of computations.
 * \param table1 One of the three tables, which won't change.
 * \param table2 One of the three tables, which won't change.
 * \param table3 One of the three tables, which won't change.
 * \note We don't need (at least yet) to have a function working for N tables, even if this
 *       function for 3 tables seems at first glance redundant with the one for 2 tables defined
 *       just above. The reason is that in order to do the union for N tables, we need to know how
 *       to do it for two. That's because we would compute for N tables using the associativity
 *       of the union : T1 U T2 U T3 ... U Tn = ((T1 U T2) U T3) ... U Tn
 *       Therefore, we need one for 2 tables anyway. And as the one for N (N>=3) would only be used
 *       (at least for now) for N=3, there is at the moment no need for such a generic union over
 *       N tables.
 */
ComputationTable UnionOfComputationTables(const ComputationTable& table1,
                                          const ComputationTable& table2,
                                          const ComputationTable& table3) {
  ComputationTable result = table1;  // Copy needed as the union of 2 writes into its first arg
  UnionOfComputationTables(&result, table2);
  UnionOfComputationTables(&result, table3);

  return result;
}

/*!
 * \brief Does the intersection of two tables of computations.
 * \param table1 One of the two tables, which won't change.
 * \param table2 The other table, which also won't change.
 */
ComputationTable IntersectComputationTables(const ComputationTable& table1,
                                            const ComputationTable& table2) {
  ComputationTable result;
  for (const auto& current : table1) {
    auto it = table2.find(current.first);
    if (it != table2.end()) {
      result[current.first] = current.second + it->second;
    }
  }
  return result;
}

/*!
 * \brief Does the intersection of three tables of computations.
 * \param table1 One of the three tables, which won't change.
 * \param table2 One of the three tables, which won't change.
 * \param table3 One of the three tables, which won't change.
 * \note We don't need (at least yet) to have a function working for N tables, even if this
 *       function for 3 tables seems at first glance redundant with the one for 2 tables defined
 *       just above. The reason is that in order to do the intersection for N tables, we need to
 *       know how to do it for two. That's because we would compute for N tables using the
 *       associativity of the intersection : T1 Inter T2 Inter T3 ... Inter Tn
 *       = ((T1 Inter T2) Inter T3) ... Inter Tn
 *       Therefore, we need one for 2 tables anyway. And as the one for N (N>=3) would only be used
 *       (at least for now) for N=3, there is at the moment no need for such a generic intersection
 *       over N tables.
 */
ComputationTable IntersectComputationTables(const ComputationTable& table1,
                                            const ComputationTable& table2,
                                            const ComputationTable& table3) {
  ComputationTable result = IntersectComputationTables(table1, table2);
  result = IntersectComputationTables(result, table3);
  return result;
}

/*!
 * \brief Recompute the number of times that each computation in table_main is seen in the tables
          contained by the vector of tables vecTables. It sets each element to the sum of the times
          it is seen in each individual table.
 * \param table_main The main table, for which we recompute the counters.
 * \param vecTables The vector of tables which won't change.
 * \note This function is needed because both the intersection (A Inter B) and the union
 *        (A U B U C) adds the individual counters found in A, B and C. So when we treat for
 *        instance an If (which contains a Cond, a Then branch and an Else branch),
 *        it will compute (Then Inter Else) U (Cond Inter Then) U (Cond Inter Else).
 *        In order to get back to the appropriate number (for instance, 3 if seen one time in each
 *        bloc), it is therefore necessary to recompute the counters afterwards, which is what this
 *        function does.
 */
void RecomputeNbTimesSeen(ComputationTable* table_main,
                          const std::vector<const ComputationTable*>& vec_tables) {
  if (table_main == nullptr) {
    return;
  }
  // For each element in the main table
  for (auto& current_elem : *table_main) {
    // We will recompute its associated counter.
    // Set its count to zero as so far it has been seen zero times
    current_elem.second = 0;
    // For each table in the vector of tables
    for (auto current_table : vec_tables) {
      // Try to find current_elem in the current table
      auto it = current_table->find(current_elem.first);
      if (it != current_table->end()) {
        // If found, increase its counter by the value found in the current table
        current_elem.second += it->second;
      }
    }
  }
}

/*!
 * \brief Builds a table for a node that has three children. A computation will be reported
   as being computed if it appears in at least two of the children, i.e. if it will aways be
   computed, regardless of the execution path.
 * \param table_child1 The table of computations done by the first child.
 * \param table_child2 The table of computations done by the second child.
 * \param table_child3 The table of computations done by the third child.
 * \note This function will be used for obtaining the computations done by If nodes and by For
 * nodes, which both have three children.
 */
ComputationTable BuildTableForThreeChildrenNode(const ComputationTable& table_child1,
                                                const ComputationTable& table_child2,
                                                const ComputationTable& table_child3) {
  ComputationTable result;
  // We look at what the children have in common
  ComputationTable child2_inter_child3 = IntersectComputationTables(table_child2, table_child3);
  ComputationTable child1_inter_child2 = IntersectComputationTables(table_child1, table_child2);
  ComputationTable child1_inter_child3 = IntersectComputationTables(table_child1, table_child3);

  // We do the union of all the things they have in common
  result = UnionOfComputationTables(child2_inter_child3, child1_inter_child2, child1_inter_child3);

  // Now we need to recompute the numbers associated with each computation, because both the
  // intersections and the union might have increased the counters, which can now be wrong.
  std::vector<const ComputationTable*> vec_tables = {&table_child1, &table_child2, &table_child3};
  RecomputeNbTimesSeen(&result, vec_tables);

  return result;
}

/*!
 * \brief Toplevel (static) method for a PrimExpr
 * \param expr The expr for which we want to know the computations done
 * \param is_eligible_computation The predicate which decides if an expression is eligible for
                                  being introduced in a new variable
 * \param can_contain_computations The predicate which decides if an expression can contain an
                                    eligible computation
 */
ComputationTable ComputationsDoneBy::GetComputationsDoneBy(
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
ComputationTable ComputationsDoneBy::GetComputationsDoneBy(
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
    UnionOfComputationTables(&table_of_computations_, it_table_expr->second);
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
    ComputationTable temp =
        ComputationsDoneByChildrenOf(expr, is_eligible_computation_, can_contain_computations_);
    // We need to do the union with `table_of_computations_` instead of just writing into it,
    // because some other childs might have added things into it too. The reason for that is
    // that `table_of_computations_` is shared between the child nodes of a given expression.
    UnionOfComputationTables(&table_of_computations_, temp);
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
    UnionOfComputationTables(&table_of_computations_, it_table_stmt->second);
    return;
  }

  // If we reach this point, it means that we have never computed before the computations done
  // by `stmt` and will do so now.

  // The computations done by a Stmt node are just the ones done by its children
  ComputationTable temp =
      ComputationsDoneByChildrenOf(stmt, is_eligible_computation_, can_contain_computations_);
  // We need to do the union with `table_of_computations_` instead of just writing into it,
  // because some other childs might have added things into it too. The reason for that is
  // that `table_of_computations_` is shared between the child nodes of a given expression.
  UnionOfComputationTables(&table_of_computations_, temp);
}

/*!
 * \brief The method which overrides the specific treatment for an IfThenElseNode
 */
void ComputationsDoneBy::VisitStmt_(const IfThenElseNode* op) {
  // We build the computations done by each of its child, but unlike the overridden method we will
  // remember each table of computations so that we can at the end compute the needed intersections

  // Calls the VisitExpr() method on the `condition` child
  VisitExpr(op->condition);
  ComputationTable computations_done_by_cond = table_of_computations_;
  // Clear it for not importing the computations of the condition in the computations of the then
  table_of_computations_.clear();

  // Then calls the VisitStmt() method on the `then_case` child
  VisitStmt(op->then_case);
  ComputationTable computations_done_by_then = table_of_computations_;
  // Clear it for not importing the computations of the then in the computations of the else
  table_of_computations_.clear();

  ComputationTable computations_done_by_else;
  if (op->else_case) {
    // And finally calls the VisitStmt() method on the `else_case` child
    VisitStmt(op->else_case.value());
    computations_done_by_else = table_of_computations_;
    table_of_computations_.clear();
  }

  // Build a table of computations for this node with three children
  table_of_computations_ = BuildTableForThreeChildrenNode(
      computations_done_by_cond, computations_done_by_then, computations_done_by_else);

  // Copy the `table_of_computations_` into the cache
  // for the future queries
  Stmt ref_to_op = GetRef<Stmt>(op);
  cache_.cache_stmt_table_computations_[ref_to_op] = table_of_computations_;
}

/*!
 * \brief The method which overrides the specific treatment for a ForNode
 */
void ComputationsDoneBy::VisitStmt_(const ForNode* op) {
  // We build the computations done by each of its child, but unlike the overridden method we will
  // remember each table of computations so that we can at the end compute the needed intersections

  // Calls the VisitExpr() method on the `min` child
  VisitExpr(op->min);
  ComputationTable computations_done_by_min = table_of_computations_;
  // Clear it for not importing the computations of the min in the computations of the extent
  table_of_computations_.clear();

  // Then calls the VisitStmt() method on the `extent` child
  VisitExpr(op->extent);
  ComputationTable computations_done_by_extent = table_of_computations_;
  // Clear it for not importing the computations of the extent in the computations of the body
  table_of_computations_.clear();

  ComputationTable computations_done_by_body;
  // And finally calls the VisitStmt() method on the `body` child
  VisitStmt(op->body);
  computations_done_by_body = table_of_computations_;
  table_of_computations_.clear();

  // Build a table of computations for this node with three children
  table_of_computations_ = BuildTableForThreeChildrenNode(
      computations_done_by_min, computations_done_by_extent, computations_done_by_body);

  // Copy the `table_of_computations_` into the cache
  // for the future queries
  Stmt ref_to_op = GetRef<Stmt>(op);
  cache_.cache_stmt_table_computations_[ref_to_op] = table_of_computations_;
}

/*!
 * \brief The method which overrides the specific treatment for a WhileNode
 */
void ComputationsDoneBy::VisitStmt_(const WhileNode* op) {
  // We build the computations done by each of its child, but unlike the overridden method we will
  // remember each table of computations so that we can at the end compute the needed intersection

  // Calls the VisitExpr() method on the `condition` child
  VisitExpr(op->condition);
  ComputationTable computations_done_by_condition = table_of_computations_;
  // Clear it for not importing the computations of the min in the computations of the extent
  table_of_computations_.clear();

  // Then calls the VisitStmt() method on the `body` child
  VisitStmt(op->body);
  ComputationTable computations_done_by_body = table_of_computations_;
  // Clear it for not importing the computations of the extent in the computations of the body
  table_of_computations_.clear();

  // Build a table of computations for this node with two children by computing what is
  // is common between the two child, i.e. computing their intersection
  table_of_computations_ =
      IntersectComputationTables(computations_done_by_condition, computations_done_by_body);

  // Copy the `table_of_computations_` into the cache
  // for the future queries
  Stmt ref_to_op = GetRef<Stmt>(op);
  cache_.cache_stmt_table_computations_[ref_to_op] = table_of_computations_;
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
ComputationTable ComputationsDoneBy::ComputationsDoneByChildrenOf(
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
ComputationTable ComputationsDoneBy::ComputationsDoneByChildrenOf(
    const Stmt& stmt, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  // We will be using an instance of the class ComputationsDoneBy for the child nodes
  // (ie, they will share the "result" that `table_of_computations_` is)
  ComputationsDoneBy computations_done_by(is_eligible_computation, can_contain_computations);
  // Calls the *dispatcher* (not the overriden method)
  computations_done_by.StmtExprVisitor::VisitStmt(stmt);
  // So now we can copy table_of_computations_ into the cache for the future queries
  // Note : in the table, the computations done by `stmt` is set to the computations done by its
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
 * \brief Print a table of computation.
 */
void PrintComputationTable(const ComputationTable& table) {
  std::cout << "{" << std::endl;
  for (const auto& current : table) {
    std::cout << "(" << current.first << ", " << current.second << ")" << std::endl;
  }
  std::cout << "}" << std::endl;
}

/*!
 * \brief Decides if two terms are equal syntactically
 */
bool EqualTerms(const PrimExpr& a, const PrimExpr& b) {
  ExprDeepEqual deep_equal_;
  return deep_equal_(a, b);
}

/*!
 * \brief Normalization function of a term, use to decide the equivalence relation of interest
 * \param expr The expression to normalize
 * \param do_normalization Whether we want the function to actually do normalization
 * \note This function can be customized
 */
PrimExpr NormalizeTerm(const PrimExpr& expr, bool do_normalization) {
  if (do_normalization) {
    // Customize here!
    // We could decide to normalize terms in a way that identifies them modulo commutativity
    // (like x+y and y+x), or modulo associativity (like (x+y)+z and x+(y+z)), etc.
    // For that, a normalization procedure (or an incomplete "pseudo-normalization" like
    // arith::Analyzer::Simplify) will be used.

    // One possible customization:
    // Here is just an attempt to do more commonings by using the pseudo-normalization function
    // offered by arith::Analyzer::Simplify(). "pseudo" because while it is correct (i.e.
    // the simplification is indeed equivalent to the original term), it is incomplete (i.e.
    // the returned term is not guaranteed to be a normal form).
    arith::Analyzer analyzer;
    return analyzer.Simplify(expr);
  } else {
    // If `do_normalization` is false, the equivalence relation just checks the syntactic equality,
    // so the normalization is just the identity function.
    return expr;
  }
}

/*!
 * \brief Decides if two terms are equivalent semantically
 */
bool EquivalentTerms(const PrimExpr& a, const PrimExpr& b, bool identify_equiv_terms) {
  // We restrict the equivalence to be decidable by a normalization procedure that is used to
  // normalize both sides, and to then compare the normal forms with the strict syntactical
  // equality
  return EqualTerms(NormalizeTerm(a, identify_equiv_terms), NormalizeTerm(b, identify_equiv_terms));
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
          Attention, the pairs returned are deterministic and will always be the same (as the same
          canonical representant will always be chosen for a given class of equivalence), but the
          order in which these pairs appear in the result is not deterministic, as it is based on
          the order in which we found items in the "normalized hashtable" `norm_table`). The caller
          is expected to sort the result anyway.
 */
std::vector<std::pair<PrimExpr, size_t>> SyntacticToSemanticComputations(
    const ComputationTable& table, bool identify_equiv_terms) {
  std::vector<std::pair<PrimExpr, size_t>> result;

  // If we do NOT identify equivalent terms, then we simply need to transform the input hashtable
  // into a vector, without doing anything else.
  if (!identify_equiv_terms) {
    // The result will contain exactly as many elements as the input `table` has
    result.reserve(table.size());
    for (const auto& elem : table) {
      result.push_back(elem);
    }

    return result;
  }

  // Otherwise, in order to identify equivalent terms, we will go through a table `norm_table`
  // where normal forms are the keys., and use it to efficiently merge equivalent terms.

  // In order to produce the result (a vector of semantical entities), the input table will be
  // normalized. This normalized table will keep the count for each set of equivalent terms
  // (i.e. each equivalence class), together with a term that did appear in this equivalence class
  // (in practice, the first term of the equivalence class that was encoutered).
  std::unordered_map<PrimExpr, std::pair<PrimExpr, size_t>, StructuralHash, ExprDeepEqual>
      norm_table;

  // In order to avoid frequent rehashing if the norm_table becomes big, we immediately ask for
  // enough space to store the amount of elements that the input table has, as it's clearly an
  // upper bound (in the worst case, each element is its own representant, and there is as many
  // equivalence classes as there are elements)
  norm_table.reserve(table.size());

  // Transform the input hashtable to a vector and sort it according to some order, as we will be
  // iterating through its items soon, and the order of appearance will be used to determine the
  // individual representant for each class of equivalence, which we want to be deterministic
  // (otherwise {x+y, y+x} could be both replaced by x+y, and on another run by y+x).
  std::vector<std::pair<PrimExpr, size_t>> sorted_items_of_table(table.begin(), table.end());

  // We do the ordering by comparing the string repr of each expr to get a determinstic ordering
  sort(sorted_items_of_table.begin(), sorted_items_of_table.end(),
       [](std::pair<PrimExpr, size_t> a, std::pair<PrimExpr, size_t> b) {
         std::stringstream a_stream;
         std::stringstream b_stream;
         a_stream << a.first;
         b_stream << b.first;
         return a_stream.str().compare(b_stream.str()) < 0;
       });

  for (const auto& elem : sorted_items_of_table) {
    PrimExpr norm_elem = NormalizeTerm(elem.first, identify_equiv_terms);
    // If the normalized term is not already a key in the normalized table
    auto it_found = norm_table.find(norm_elem);
    if (it_found == norm_table.end()) {
      // Then we add the mapping `norm_elem` -> (`elem`.first, `elem`.second) to the norm table
      // (i.e. `norm_elem` has been seen `elem`.second many times so far, and the chosen element
      // to represent the equivalence class will be `elem`.first as it's the first element of the
      // class that we see)
      norm_table[norm_elem] = elem;
    } else {
      // Otherwise, it's not the first time we see a term in this equivalence class, so we just
      // increase the count of this equivalence class as we now have `elem`.second additional items
      // coming to the equivalence class.
      it_found->second.second += elem.second;
    }
  }

  // norm_table.size() is the number of equivalence class that we have built, so it's exactly the
  // number of items that we will return in the vector of semantical entities
  result.reserve(norm_table.size());

  // Transform the intermediate hashtable `norm_table` into a vector, forgetting the keys,
  // (which are the normal forms), as they won't be used as the canonical representants (which are
  // instead the first element of each class that is effectively seen)
  // Careful : the pairs will never change (the canonical represantants chosen will always be the
  // same), but the order in which the pairs are produced can vary as we are iterating through the
  // hashtable `norm_table`. It is not an issue as the called will be sorting the result anyway.
  std::unordered_map<PrimExpr, std::pair<PrimExpr, size_t>, StructuralHash,
                     ExprDeepEqual>::const_iterator it_norm_table;
  for (it_norm_table = norm_table.begin(); it_norm_table != norm_table.end(); ++it_norm_table) {
    result.push_back(it_norm_table->second);
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
  // as soon as two occurrences are seen, but a smaller one could need three or more occurrences
  // for being introduced in a variable.

  // But for now, we factorize any eligible item that we see at least twice, regardless of its size
  return nb_times_seen >= 2;
}

/*!
 * \brief Inserts a pair (expr,nb) to a sorted vector of such pairs (which is sorted by decreasing
          size of expressions) and maintain the vector sorted while doing so.
 */
void InsertElemToSortedSemanticComputations(std::vector<std::pair<PrimExpr, size_t>>* sorted_vec,
                                            const std::pair<PrimExpr, size_t>& pair) {
  if (sorted_vec == nullptr) {
    return;
  }
  // Find the insertion point using std::lower_bound on a comparison that uses
  // CalculateExprComplexity(), which computes the "size" of an expr.
  // std::lower_boud returns an iterator pointing to the first element on which the comparison
  // does not return true with the given value (`pair` here), i.e, an iterator pointing to the
  // first element that is not greater or equal than `pair`, i.e, the first element that is
  // strictly smaller than `pair`.
  auto insertion_point = std::lower_bound(
      sorted_vec->begin(), sorted_vec->end(), pair,
      [](const std::pair<PrimExpr, size_t>& left, const std::pair<PrimExpr, size_t>& right) {
        return (CalculateExprComplexity(left.first) >= CalculateExprComplexity(right.first));
      });
  sorted_vec->insert(insertion_point, pair);
}

/*!
 * \brief Inserts a vector of expressions into a sorted vector of computations (which is sorted by
          decreasing size of the expression) and maintain the vector sorted while doing so.
 */
void InsertVectorToSortedSemanticComputations(std::vector<std::pair<PrimExpr, size_t>>* sorted_vec,
                                              const std::vector<PrimExpr>& vec_to_add,
                                              bool identify_equiv_terms) {
  if (sorted_vec == nullptr) {
    return;
  }
  for (auto elem_to_add : vec_to_add) {
    // See if the current element to add (or an equivalent one) is already present
    // in the sorted vector
    auto it_found =
        std::find_if(sorted_vec->begin(), sorted_vec->end(),
                     [elem_to_add, identify_equiv_terms](std::pair<PrimExpr, size_t> elem) {
                       return EquivalentTerms(elem.first, elem_to_add, identify_equiv_terms);
                     });

    // If we found `elem_to_add` (or an equivalent expression) already in sorted_vec
    if (it_found != sorted_vec->end()) {
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
