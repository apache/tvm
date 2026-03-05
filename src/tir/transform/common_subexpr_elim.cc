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
 * \file common_subexpr_elim.cc
 * \brief Implementation of the Common Subexpressions Elimination (CSE) pass
           which rewrites statements and expressions in order to eliminate
           redundant computations. In order to achieve that, common (sub-)
           expressions are introduced into variables with let-in bindings,
           and the places where the expression was used are replaced with
           the freshly introduced variable.
 */

#include "common_subexpr_elim.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/transform.h>  // For the class Pass and the class PassContext
#include <tvm/tir/analysis.h>  // For the analysis which gives the size of an expr
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>  // For the class PrimFunc
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>  // For the decl of the function returning the pass

#include <algorithm>  // For the algorithm std::find
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../analysis/check_contains.h"  // For the visitor CheckContains
#include "common_subexpr_elim_tools.h"   // For the auxiliary analysis (visitors) and tools
#include "replace_selected_expr.h"       // For the mutator ReplaceSelectedExpr

namespace tvm {
namespace tir {

/*!
 * \brief Check whether a computation is forbidden for being treated by the CSE pass.
          The important thing about forbidden computations is that not only we won't want
          to collect them for the CSE pass, but we also won't even want to collect computations
          that contain them.
          The reason is that reusing such computations would change the semantics of the program,
          and therefore before doing any introduction of var or any reuse of already introduced
          variables, we will make sure that the computation being considered is not forbidden, and
          that it does not even contain a forbidden computation.
 * \param expr The expression to check
 * \return Whether `expr` is a forbidden computation or not
 */
bool CommonSubexpressionEliminator::ForbiddenComputation(const PrimExpr& expr) {
  // Function calls, loads and buffer loads are absolutely forbidden as introducing them into
  // variables would change the semantics of the program.
  return (expr.as<CallNode>() != nullptr || expr.as<BufferLoadNode>() != nullptr);
}

/*!
 * \brief Predicate used for verifying that a computation is eligible for being treated by
          the CSE pass, i.e. for being introduced into a variable / for being replaced by a
          variable.
          Being eligible is a conjunction of a few conditions, like not being an atom (constant
          or variable), not being a forbidden node, not containing a forbidden node, etc.
 * \param expr The expression to check
 * \return Whether `expr` is an eligible computation or not
 */
bool CommonSubexpressionEliminator::IsEligibleComputation(const PrimExpr& expr) {
  return (
      // In order to be eligible, the given expression should not be a constant
      (expr.as<IntImmNode>() == nullptr) && (expr.as<FloatImmNode>() == nullptr) &&
      (expr.as<StringImmNode>() == nullptr)
      // and it should not be a variable
      && (expr.as<VarNode>() == nullptr)
      // and it should not be a forbidden computation (function calls and loads)
      && (!ForbiddenComputation(expr))
      // and it should not even contain a forbidden computation (function calls and loads)
      // the reason is that we don't want to register expressions like (x + f(y)) or
      // (x + Mem[i]) as introducing them into variables could change the semantics
      && (!CheckContains::ExprContains(expr, ForbiddenComputation))
      // and it should not be a ramp node or a broadcast node due to some internals TVM
      // constraints (which check for these node explicitely without performing any
      // evaluation first, so if they have been put into variables it fails)
      && (expr.as<RampNode>() == nullptr) && (expr.as<BroadcastNode>() == nullptr));
}

/*!
 * \brief Predicate used (when considering eligible computations) for only diving into
          expressions that are allowed to contain eligible computations. Customize this predicate
          if you want to make it forbidden to rewrite inside a specific node, like inside
          a Load node for instance.
 * \param expr The expression to check
 * \return Whether `expr` can contain some eligible computations or not, and therefore
             if recursing inside `expr` is necessary.
 */
bool CommonSubexpressionEliminator::CanContainEligibleComputations(const PrimExpr& expr) {
  // Uncomment the next line to prevent the collection and the replacement of eligible computations
  // inside the index of Load nodes. We initially thought that this would be needed in order to
  // not harm the indexing mode of the CPU, but as we are still far from ASM code, we
  // finally want to perform such simplifications, which tend to happen fairly frequently.

  // return (expr.as<BufferLoadNode>() == nullptr)
  return true;
}

/*!
 * \brief Implements an order on pairs (expression,frequency). First attempts to compare them
          using the size of the expression. If it is the same, decides something else still
          deterministic.
 * \param a The first pair
 * \param b The second pair
 * \return A boolean telling if the first pair `a` comes before the second pair `b`
 * \note We need this order to be deterministic in order to have a fully deterministic pass,
 *       as we will deal with elements that are coming from a hashtable, but the order in which
 *       they appeared in the hashtable was based on some runtime addresses, so it can potentially
 *       change with every execution.
 */
bool CommonSubexpressionEliminator::OrderOnExprAndFrequency(const std::pair<PrimExpr, size_t>& a,
                                                            const std::pair<PrimExpr, size_t>& b) {
  size_t a_size = CalculateExprComplexity(a.first);
  size_t b_size = CalculateExprComplexity(b.first);
  return a_size > b_size;
}

/*!
 * \brief Generates a new fresh variable, whose name will be cse_vi.
 * \param type_annotation The type of the new variable to generate
 * \return A new variable of type `type_annotation` called cse_vi where i is the first available
            integer.
 */
Var CommonSubexpressionEliminator::GenerateNewVar(DataType type_annotation) {
  // Increase `num_last_try_` for this new attempt
  num_last_try_++;
  // Builds the variable name, which is cse_vi where i will go up from 1
  std::string prefix = "cse_v";
  std::string name = prefix.append(std::to_string(num_last_try_));
  // Builds a ffi::String using the std::string
  ffi::String string_name(name);

  // Check that the name that we want to use for the new variable isn't already being used
  // (names don't really have to be unique as they are just hints, and having the same name
  // doesn't means that it's the same variable, but it's clearer for dumps)
  if (UsesVarName::StmtUsesVarName(initial_body_, string_name)) {
    // If the name is already used, call ourselves recursively for trying with the next one
    return GenerateNewVar(type_annotation);
  }

  // Increase `nb_var_` for this new generation of variable that we have just done
  nb_var_++;

  // Return a new Variable using the name built and the given type_annotation
  return (Var(string_name, type_annotation));
}

/*!
 * \brief Gives the number of variables generated by the CSE on the current function
           (i.e., getter for `nb_var_`).
 * \return A copy of `nb_var_`
 */
int CommonSubexpressionEliminator::GetNbVarGenerated() { return nb_var_; }

/*!
 * \brief Toplevel (static) method that performs Common Subexpression Elimination on
          a given statement (which should be the body of a PrimFunc). This method should be
          called for each PrimFunc definition.
 * \param stmt The statement of the function being analyzed, on which we want to perform CSE
 * \param context_init The initial context, which should contain the formal parameters
                          of the function being analyzed
 * \return A new statement where CSE has been performed
 */
Stmt CommonSubexpressionEliminator::PerformCSE(const Stmt& stmt, const Context& context_init,
                                               bool identify_equiv_terms) {
  // As this function is being called for each PrimFunc definition, we create a new instance
  // for the one we are having now.
  CommonSubexpressionEliminator common_subexpression_eliminator(stmt, context_init,
                                                                identify_equiv_terms);
  return common_subexpression_eliminator.VisitStmt(stmt);
}

/*!
 * \brief Protected constructor of CommonSubexpressionEliminator.
 * \param context_init The context at the beginning of the CSE pass. It should contain the
                        formal parameters of the function that will be analyzed
 */
CommonSubexpressionEliminator::CommonSubexpressionEliminator(const Stmt& stmt,
                                                             const Context& context_init,
                                                             bool identify_equiv_terms)
    : initial_body_(stmt), context_(context_init), identify_equiv_terms_(identify_equiv_terms) {
  // The initial scope level (from ScopeStack's constructor) does not need
  // EnterContextScope() because it should never be popped -- it persists
  // for the lifetime of the CSE pass and holds the function parameters.
}

/*!
 * \brief The method which overrides the generic dispatcher of StmtExprMutator.
          Entry point to the common subexpression elimination mutator for expressions.
 * \param expr The expression to mutate
 */
PrimExpr CommonSubexpressionEliminator::VisitExpr(const PrimExpr& expr) {
  bool variables_created = false;  // Will be needed for knowing if the CSE has created new vars
  PrimExpr result = expr;

  // Obtain the (syntactic) eligible computations done by the input expression, and keep it as
  // a ComputationTable, which is a mapping from PrimExpr to size_t, where the size_t is the
  // number of time this exact syntactic computation is being computed.
  ComputationTable table_syntactic_comp_done_by_expr = ComputationsDoneBy::GetComputationsDoneBy(
      expr, IsEligibleComputation, CanContainEligibleComputations);

  // Transform the hashtable of *syntactic* eligible computations into a vector of pairs
  // containing *semantic* entities, i.e. where equivalent computations are merged.
  std::vector<std::pair<PrimExpr, size_t>> semantic_comp_done_by_expr =
      SyntacticToSemanticComputations(table_syntactic_comp_done_by_expr, identify_equiv_terms_);

  // Sort the vector of semantic entities by decreasing size
  std::stable_sort(semantic_comp_done_by_expr.begin(), semantic_comp_done_by_expr.end(),
                   OrderOnExprAndFrequency);

  // For each computation done (considering them from biggest to smallest)
  for (size_t i = 0; i < semantic_comp_done_by_expr.size(); i++) {
    std::pair<PrimExpr, size_t>& computation_and_nb = semantic_comp_done_by_expr[i];

    bool ident_equiv_terms = identify_equiv_terms_;  // To avoid the capture of "this"

    // The predicate later used (when doing replacements) to select expressions that are
    // equivalent to the current computation (`computation_and_nb.first`)
    std::function<bool(const PrimExpr&)> predicate_selector =
        [computation_and_nb, ident_equiv_terms](const PrimExpr& current_expr) {
          // `current_expr` should be equivalent to `computation_and_nb.first`, but we also check
          // that `current_expr` is an eligible computation even if we know that
          // `computation_and_nb.first` is eligible by construction, in case that one day the
          // equivalence relation would not preserve the eligibility any more (even though that
          // would probably be a very weird equivalence).
          return (EquivalentTerms(current_expr, computation_and_nb.first, ident_equiv_terms) &&
                  IsEligibleComputation(current_expr));
        };

    // See if there is a pair (`var`, `value`) in the context where `value` is semantically
    // equivalent to `computation_and_nb.first`
    auto it_on_var = std::find_if(
        context_.begin(), context_.end(),
        [computation_and_nb, ident_equiv_terms](const std::pair<Var, MaybeValue>& var_and_value) {
          // Note : safe to call value() as we check has_value() just before
          return (var_and_value.second.has_value() &&
                  EquivalentTerms(var_and_value.second.value(), computation_and_nb.first,
                                  ident_equiv_terms));
        });

    // Case where we have a perfectly equivalent computation already available in a variable
    // introduced (i.e, present in context_).
    // Note that this case is needed when the user has written something like
    // [let x = A in ....A...A...] : we need to be able to replace all the occurrences of A by
    // an already existing variable holding A, when such a variable happens to exist.
    if (it_on_var != context_.end()) {
      // Replace in the current `result` everything that is selected by the selector with
      // the existing variable, without diving into expressions in which we don't have the
      // right to dive.
      result = ReplaceSelectedExpr::ReplaceSelectedExprInExpr(
          result, predicate_selector, it_on_var->first, CanContainEligibleComputations);
    } else {
      // The current computation is not equivalent to a computation already done. We will
      // need to see if we want to introduce it.

      // --- Chunk needed for reusing the UndefinedVars() analysis ---
      // 1 - Wraps the computation into a statement
      Stmt computation_wrapped_in_stmt = Evaluate(computation_and_nb.first);
      // 2.1 - Transform the context into a vector of variables instead of pairs
      std::function<Var(const std::pair<Var, MaybeValue>&)> forget_value =
          [](const std::pair<Var, MaybeValue>& pair) { return pair.first; };
      std::vector<Var> vector_vars_known = VectorMap(context_, forget_value);
      // 2.2 - Transform the std::vector into an Array
      ffi::Array<Var> array_vars_known = ffi::Array<Var>(vector_vars_known);
      // --- End of chunk needed for reusing the UndefinedVars() analysis ---

      // We use the UndefinedVars() analysis to get the undefined vars of the computation
      ffi::Array<Var> vars_undefined = UndefinedVars(computation_wrapped_in_stmt, array_vars_known);

      // Check if we can introduce it : if it contains no undefined variables and if we want
      // to introduce it according to the predicate
      if (vars_undefined.empty() &&
          PredicateIntroVarForComputation(computation_and_nb.first, computation_and_nb.second)) {
        // Create a new variable for this computation
        Var new_var = GenerateNewVar(computation_and_nb.first.dtype());
        // Replace in the current `result` everything that is selected by the selector with
        // the new variable, without diving into expressions in which we don't have the
        // right to dive.
        result = ReplaceSelectedExpr::ReplaceSelectedExprInExpr(result, predicate_selector, new_var,
                                                                CanContainEligibleComputations);
        // Build a let-in that introduces the new variable in the current `result`
        result = Let(new_var, computation_and_nb.first, result);
        // We don't add the variable to the context because the invariant is that the
        // context is the context in which 'result' makes sense, and we've just updated it.
      } else {
        // Here it's not doable to introduce (via a let-in) the computation at this level
        // as it contains variables that are not yet declared, and/or because the predicate
        // did not select it.
        // Either way, we will simply add to the vector of computations the direct subexprs
        // of the current computation, as these ones might be good candidates
        // for being introduced into variables.
        // Note that we don't need to add all of its subexpressions, but only its *direct*
        // subexpressions as we consider them from biggest to smallest, and if they were
        // all added at once, then there could be dependencies between them, as commoning
        // one of them could remove some other possibilities.

        // Computing the direct subexpressions will return a small number of direct
        // subexpressions (typically 0 to 3)
        std::vector<PrimExpr> direct_subexprs = DirectSubexpr::GetDirectSubexpressions(
            computation_and_nb.first, IsEligibleComputation, CanContainEligibleComputations);
        // The following insertion will maintain `semantic_comp_done_by_expr` sorted (by
        // decreasing size/complexity), and it will only insert at locations > i as the
        // direct subexprs are necessarily smaller than the current computation.
        InsertVectorToSortedSemanticComputations(&semantic_comp_done_by_expr, direct_subexprs,
                                                 identify_equiv_terms_);
      }
    }
    // Note : we do not remove the current element, as we never look back in the local vector
  }  // End of for loop

  // If the CSE pass has created some variables, then we run it again as more commoning could
  // potentially happen using the new variables introduced
  if (variables_created) {
    result = VisitExpr(result);
  } else {
    // But if no changes were performed, we recurse inside the children by calling the dispatcher.
    // Calling the dispatcher to the specific treatments, which will update the context
    // appropriately before doing the recursive calls on the children nodes
    result = StmtExprMutator::VisitExpr(result);
  }

  return result;
}

/*!
 * \brief The method which overrides the specific treatment for a LetNode.
 *
 * The let-in expression introduces a new variable binding that is only visible
 * within the body. We use context_scope_.WithNewScope to automatically clean up
 * the binding when the body has been visited, replacing the old manual
 * save/restore of context_.
 */
PrimExpr CommonSubexpressionEliminator::VisitExpr_(const LetNode* op) {
  // At this point, we have already done the generic treatment of introducing (via let-in) what
  // was doable at the toplevel of the given let-in.

  // Recurse on the `value` field for potentially rewriting it
  PrimExpr value_new = VisitExpr(op->value);

  // Visit the body in a new scope. The let-in variable binding is added to the
  // context inside the scope and automatically removed when the scope exits.
  PrimExpr body_new = context_scope_.WithNewScope([&]() -> PrimExpr {
    EnterContextScope();
    // Augment the context with the association (`var`, `value`) for the body
    context_.push_back({op->var, MaybeValue(op->value)});
    // Recurse on the `body` (with this extended context)
    // The recursive call will have potentially done new simplifications, because in this recursive
    // call `var` will be a part of the context.
    return VisitExpr(op->body);
  });

  // Rebuild the let-in with a new `value_new` and `body_new` where new simplifications might
  // have been done.

  // If the `value` and the `body` of the let-in have been rewritten to the same thing
  if (value_new.same_as(op->value) && body_new.same_as(op->body)) {
    // then return a reference to the same node
    return ffi::GetRef<PrimExpr>(op);
  } else {
    // Otherwise return a let-in built with the new `value_new` and the new `body_new` that
    // have just been obtained
    return Let(op->var, value_new, body_new, op->span);
  }
}

/*!
 * \brief The method which overrides the generic dispatcher of StmtExprMutator.
          Entry point to the common subexpression elimination mutator for statements.
 * \param stmt The statement to mutate.
 */
Stmt CommonSubexpressionEliminator::VisitStmt(const Stmt& stmt) {
  bool variables_created = false;  // Will be needed for knowing if the CSE has created new vars
  Stmt result = stmt;

  // Obtain the (syntactic) eligible computations done by the input statement, and keep it as
  // a ComputationTable, which is a mapping from PrimExpr to size_t, where the size_t is the
  // number of time this exact syntactic computation is being computed.
  ComputationTable table_syntactic_comp_done_by_stmt = ComputationsDoneBy::GetComputationsDoneBy(
      stmt, IsEligibleComputation, CanContainEligibleComputations);

  // Transform the hashtable of *syntactic* eligible computations into a vector of pairs
  // containing *semantic* entities, i.e. where equivalent computations are merged.
  std::vector<std::pair<PrimExpr, size_t>> semantic_comp_done_by_stmt =
      SyntacticToSemanticComputations(table_syntactic_comp_done_by_stmt, identify_equiv_terms_);

  // Sort the vector of semantic entities by decreasing size
  std::stable_sort(semantic_comp_done_by_stmt.begin(), semantic_comp_done_by_stmt.end(),
                   OrderOnExprAndFrequency);

  // For each computation done (considering them from biggest to smallest)
  for (size_t i = 0; i < semantic_comp_done_by_stmt.size(); i++) {
    std::pair<PrimExpr, size_t>& computation_and_nb = semantic_comp_done_by_stmt[i];

    bool ident_equiv_terms = identify_equiv_terms_;  // To avoid the capture of "this"

    // The predicate later used (when doing replacements) to select expressions that are
    // equivalent to the current computation (`computation_and_nb.first`)
    std::function<bool(const PrimExpr&)> predicate_selector =
        [computation_and_nb, ident_equiv_terms](const PrimExpr& current_expr) {
          // `current_expr` should be equivalent to `computation_and_nb.first`, but we also check
          // that `current_expr` is an eligible computation even if we know that
          // `computation_and_nb.first` is eligible by construction, in case that one day the
          // equivalence relation would not preserve the eligibility any more (even though that
          // would probably be a very weird equivalence).
          return (EquivalentTerms(current_expr, computation_and_nb.first, ident_equiv_terms) &&
                  IsEligibleComputation(current_expr));
        };

    // See if there is a pair (`var`, `value`) in the context where `value` is semantically
    // equivalent to `computation_and_nb.first`
    auto it_on_var = std::find_if(
        context_.begin(), context_.end(),
        [computation_and_nb, ident_equiv_terms](const std::pair<Var, MaybeValue>& var_and_value) {
          // Note : safe to call value() as we check has_value() just before
          return (var_and_value.second.has_value() &&
                  EquivalentTerms(var_and_value.second.value(), computation_and_nb.first,
                                  ident_equiv_terms));
        });

    // Case where we have a perfectly equivalent computation already available in a variable
    // introduced (i.e, present in context_).
    // Note that this case is needed when the user has written something like
    // [let x = A in ....A...A...] : we need to be able to replace all the occurrences of A by
    // an already existing variable holding A, when such a variable happens to exist.
    if (it_on_var != context_.end()) {
      // Replace in the current `result` everything that is selected by the selector with
      // the existing variable, without diving into expressions in which we don't have the
      // right to dive.
      result = ReplaceSelectedExpr::ReplaceSelectedExprInStmt(
          result, predicate_selector, it_on_var->first, CanContainEligibleComputations);
    } else {
      // The current computation is not equivalent to a computation already done. We will
      // need to see if we want to introduce it.

      // --- Chunk needed for reusing the UndefinedVars() analysis ---
      // 1 - Wraps the computation into a statement
      Stmt computation_wrapped_in_stmt = Evaluate(computation_and_nb.first);
      // 2.1 - Transform the context into a vector of variables instead of pairs
      std::function<Var(const std::pair<Var, MaybeValue>&)> forget_value =
          [](const std::pair<Var, MaybeValue>& pair) { return pair.first; };
      std::vector<Var> vector_vars_known = VectorMap(context_, forget_value);
      // 2.2 - Transform the std::vector into an Array
      ffi::Array<Var> array_vars_known = ffi::Array<Var>(vector_vars_known);
      // --- End of chunk needed for reusing the UndefinedVars() analysis ---

      // We use the UndefinedVars() analysis to get the undefined vars of the computation
      ffi::Array<Var> vars_undefined = UndefinedVars(computation_wrapped_in_stmt, array_vars_known);

      // Check if we can introduce it : if it contains no undefined variables and if we want
      // to introduce it according to the predicate
      if (vars_undefined.empty() &&
          PredicateIntroVarForComputation(computation_and_nb.first, computation_and_nb.second)) {
        // Create a new variable for this computation
        Var new_var = GenerateNewVar(computation_and_nb.first.dtype());
        variables_created = true;
        // Replace in the current `result` everything that is selected by the selector with
        // the new variable, without diving into expressions in which we don't have the
        // right to dive.
        result = ReplaceSelectedExpr::ReplaceSelectedExprInStmt(result, predicate_selector, new_var,
                                                                CanContainEligibleComputations);
        // Build a bind that introduces the new variable before the current `result`
        result = SeqStmt({Bind(new_var, computation_and_nb.first), result});
        // We don't add the variable to the context because the invariant is that the
        // context is the context in which 'result' makes sense, and we've just updated it.
      } else {
        // Here it's not doable to introduce (via a let-in) the computation at this level
        // as it contains variables that are not yet declared, and/or because the predicate
        // did not select it.
        // Either way, we will simply add to the vector of computations the direct subexprs
        // of the current computation, as these ones might be good candidates
        // for being introduced into variables.
        // Note that we don't need to add all of its subexpressions, but only its *direct*
        // subexpressions as we consider them from biggest to smallest, and if they were
        // all added at once, then there could be dependencies between them, as commoning
        // one of them could remove some other possibilities.

        // Computing the direct subexpressions will return a small number of direct
        // subexpressions (typically 0 to 3)
        std::vector<PrimExpr> direct_subexprs = DirectSubexpr::GetDirectSubexpressions(
            computation_and_nb.first, IsEligibleComputation, CanContainEligibleComputations);
        // The following insertion will maintain `semantic_comp_done_by_stmt` sorted (by
        // decreasing size/complexity), and it will only insert at locations > i as the
        // direct subexprs are necessarily smaller than the current computation.
        InsertVectorToSortedSemanticComputations(&semantic_comp_done_by_stmt, direct_subexprs,
                                                 identify_equiv_terms_);
      }
    }
    // Note : we do not remove the current element, as we never look back in the local vector
  }  // End of for loop

  // If the CSE pass has created some variables, then we run it again as more commoning could
  // potentially happen using the new variables introduced
  if (variables_created) {
    result = VisitStmt(result);
  } else {
    // But if no changes were performed, we recurse inside the children by calling the dispatcher.
    // Calling the dispatcher to the specific treatments, which will update the context
    // appropriately before doing the recursive calls on the children nodes
    result = StmtExprMutator::VisitStmt(result);
  }

  return result;
}

/*!
 * \brief The method which overrides the specific treatment for a BindNode.
 *
 * BindNode adds a (var, value) entry to the flat context_ vector. This entry
 * persists across subsequent SeqStmt siblings in the same scope, enabling CSE
 * to find common subexpressions that reference bind-defined variables.
 * Cleanup happens automatically when the enclosing body-carrying statement's
 * scope exits (via ContextScopeLevel's destructor), so no manual save/restore
 * is needed here.
 */
Stmt CommonSubexpressionEliminator::VisitStmt_(const BindNode* op) {
  // Recurse on the `value` field for potentially rewriting it
  PrimExpr value_new = VisitExpr(op->value);

  // Augment the context with the association (`var`, `value`).
  // This persists across SeqStmt siblings and is cleaned up by the
  // enclosing scope's ContextScopeLevel destructor.
  context_.push_back({op->var, MaybeValue(op->value)});

  // Rebuild the Bind if value changed
  if (value_new.same_as(op->value)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    return Bind(op->var, value_new, op->span);
  }
}

/*!
 * \brief Whether a Bind value is trivial (constant or variable), meaning it cannot
 * contribute eligible computations for CSE and can be safely batched.
 */
static bool IsTrivialBindValue(const PrimExpr& value) {
  return value.as<IntImmNode>() != nullptr || value.as<FloatImmNode>() != nullptr ||
         value.as<StringImmNode>() != nullptr || value.as<VarNode>() != nullptr;
}

/*!
 * \brief The method which overrides the specific treatment for a SeqStmtNode.
 *
 * Processes the flat sequence using a hybrid strategy that avoids the O(n^2)
 * complexity of wrapping remaining siblings after every single Bind node:
 *
 * - Trivial Bind nodes (constant/variable values) are batched: their values
 *   are visited via VisitExpr, context_ is augmented, but the expensive
 *   cross-sibling CSE is deferred until the batch ends.
 * - Non-trivial Bind nodes (whose values may contain eligible computations)
 *   use the wrap-remaining-siblings pattern to enable cross-sibling CSE.
 * - After any Bind (trivial batch end or non-trivial), remaining siblings are
 *   wrapped into a body and VisitStmt is called once for cross-sibling CSE.
 * - Non-Bind children are visited individually via VisitStmt.
 *
 * This reduces the common case of many consecutive trivial Binds (e.g., variable
 * definitions with constant values) from O(n^2) to O(n), while preserving full
 * CSE effectiveness for non-trivial Bind values.
 *
 * Context cleanup is handled automatically by ScopeStack.
 */
Stmt CommonSubexpressionEliminator::VisitStmt_(const SeqStmtNode* op) {
  ffi::Array<Stmt> new_seq;
  size_t i = 0;

  while (i < op->seq.size()) {
    if (auto* bind = op->seq[i].as<BindNode>()) {
      // Batch consecutive trivial Bind nodes (constant/variable values).
      // These can't contribute common subexpressions, so it's safe to defer
      // the cross-sibling CSE until the entire batch is processed.
      if (IsTrivialBindValue(bind->value)) {
        while (i < op->seq.size()) {
          auto* b = op->seq[i].as<BindNode>();
          if (!b || !IsTrivialBindValue(b->value)) break;
          PrimExpr value_new = VisitExpr(b->value);
          context_.push_back({b->var, MaybeValue(b->value)});
          Stmt bind_new =
              value_new.same_as(b->value) ? ffi::GetRef<Stmt>(b) : Bind(b->var, value_new, b->span);
          new_seq.push_back(bind_new);
          ++i;
        }
      } else {
        // Non-trivial Bind: visit value, augment context, then wrap remaining
        // siblings and call VisitStmt for cross-sibling CSE.
        PrimExpr value_new = VisitExpr(bind->value);
        context_.push_back({bind->var, MaybeValue(bind->value)});
        Stmt bind_new = value_new.same_as(bind->value) ? ffi::GetRef<Stmt>(bind)
                                                       : Bind(bind->var, value_new, bind->span);
        new_seq.push_back(bind_new);
        ++i;
      }
      // After the Bind (batch or single), wrap remaining siblings [i..end) and
      // call VisitStmt once for cross-sibling CSE with the updated context.
      if (i < op->seq.size()) {
        Stmt body;
        if (i + 1 == op->seq.size()) {
          body = op->seq[i];
        } else {
          ffi::Array<Stmt> rest;
          for (size_t j = i; j < op->seq.size(); ++j) rest.push_back(op->seq[j]);
          body = SeqStmt(rest);
        }
        Stmt body_new = VisitStmt(body);
        // Flatten the result.
        if (auto* inner = body_new.as<SeqStmtNode>()) {
          for (const auto& s : inner->seq) new_seq.push_back(s);
        } else {
          new_seq.push_back(body_new);
        }
        return SeqStmt::Flatten(new_seq);
      }
    } else {
      // Non-Bind child: visit individually via VisitStmt.
      Stmt child_new = VisitStmt(op->seq[i]);
      if (auto* inner = child_new.as<SeqStmtNode>()) {
        for (const auto& s : inner->seq) new_seq.push_back(s);
      } else {
        new_seq.push_back(child_new);
      }
      ++i;
    }
  }

  return SeqStmt::Flatten(new_seq);
}

/*!
 * \brief The method which overrides the specific treatment for a ForNode.
 *
 * The for loop introduces a loop variable that is only visible within the body.
 * We use context_scope_.WithNewScope to create a scope boundary: the loop
 * variable (with no value, since it changes each iteration) is pushed inside
 * the scope and automatically cleaned up on exit, replacing the old manual
 * save/restore of context_.
 */
Stmt CommonSubexpressionEliminator::VisitStmt_(const ForNode* op) {
  // At this point, we have already done the generic treatment of introducing (via let-in) what
  // was doable at the toplevel of the given for loop.

  // Recurse on the `min` field for potentially rewriting it
  PrimExpr min_new = VisitExpr(op->min);

  // Recurse on the `extent` field for potentially rewriting it
  PrimExpr extent_new = VisitExpr(op->extent);

  // Visit the body in a new scope. The loop variable is added to context_ inside
  // the scope and automatically removed when the scope exits.
  Stmt body_new = context_scope_.WithNewScope([&]() -> Stmt {
    EnterContextScope();
    // Add loop_var with no value (its value changes each iteration)
    context_.push_back({op->loop_var, MaybeValue()});
    return VisitStmt(op->body);
  });

  // Rebuild the for loop with (potentially) a new `min_new`, `extent_new` and `body_new`, where
  // new simplifications might have been done.

  // If the `min`, `extent` and `body` of the for loop have been rewritten to the same thing
  if (min_new.same_as(op->min) && extent_new.same_as(op->extent) && body_new.same_as(op->body)) {
    // Return a reference to the same node
    return ffi::GetRef<Stmt>(op);
  } else {
    // Otherwise return a for node built with the new `min_new`, `extent_new` and `body_new`
    // that have just been obtained
    return For(op->loop_var, min_new, extent_new, op->kind, body_new, op->thread_binding,
               op->annotations, op->step, op->span);
  }
}

/*!
 * \brief The method which overrides the specific treatment for an IfThenElseNode.
 *
 * Each branch of the if-then-else gets its own scope, preventing context entries
 * (e.g., from Bind nodes inside one branch) from leaking into the other branch.
 * Without this override, the default StmtExprMutator would visit both branches
 * in the same scope, which could cause incorrect CSE across branches.
 */
Stmt CommonSubexpressionEliminator::VisitStmt_(const IfThenElseNode* op) {
  PrimExpr condition_new = VisitExpr(op->condition);

  // Each branch gets its own scope to prevent context leaks between branches
  Stmt then_new = context_scope_.WithNewScope([&]() -> Stmt {
    EnterContextScope();
    return VisitStmt(op->then_case);
  });

  ffi::Optional<Stmt> else_new;
  if (op->else_case) {
    else_new = context_scope_.WithNewScope([&]() -> Stmt {
      EnterContextScope();
      return VisitStmt(op->else_case.value());
    });
  }

  if (condition_new.same_as(op->condition) && then_new.same_as(op->then_case) &&
      else_new.same_as(op->else_case)) {
    return ffi::GetRef<Stmt>(op);
  }
  return IfThenElse(condition_new, then_new, else_new, op->span);
}

/*!
 * \brief The method which overrides the specific treatment for an AttrStmtNode.
 *
 * AttrStmt has a body that may contain Bind nodes. A scope boundary prevents
 * context entries from the body from leaking to subsequent statements.
 */
Stmt CommonSubexpressionEliminator::VisitStmt_(const AttrStmtNode* op) {
  PrimExpr value_new = VisitExpr(op->value);

  // The body gets its own scope to contain any context entries added within it
  Stmt body_new = context_scope_.WithNewScope([&]() -> Stmt {
    EnterContextScope();
    return VisitStmt(op->body);
  });

  if (value_new.same_as(op->value) && body_new.same_as(op->body)) {
    return ffi::GetRef<Stmt>(op);
  }
  return AttrStmt(op->node, op->attr_key, value_new, body_new, op->span);
}

/*!
 * \brief The method which overrides the specific treatment for an AllocBufferNode.
 *
 * AllocBuffer has a body and introduces a buffer. A scope boundary
 * prevents context entries from the body from leaking outward.
 */
Stmt CommonSubexpressionEliminator::VisitStmt_(const AllocBufferNode* op) {
  // With flat semantics, context entries persist to subsequent siblings.
  return StmtExprMutator::VisitStmt_(op);
}

/*!
 * \brief The method which overrides the specific treatment for a DeclBufferNode.
 *
 * With flat semantics, DeclBuffer has no body. Context entries persist
 * to subsequent siblings in the enclosing scope (same as Bind).
 */
Stmt CommonSubexpressionEliminator::VisitStmt_(const DeclBufferNode* op) {
  return StmtExprMutator::VisitStmt_(op);
}

/*!
 * \brief The method which overrides the specific treatment for a WhileNode.
 *
 * While loop has a body that may contain Bind nodes. A scope boundary prevents
 * context entries from the body from leaking outward.
 */
Stmt CommonSubexpressionEliminator::VisitStmt_(const WhileNode* op) {
  PrimExpr condition_new = VisitExpr(op->condition);

  // The body gets its own scope to contain any context entries added within it
  Stmt body_new = context_scope_.WithNewScope([&]() -> Stmt {
    EnterContextScope();
    return VisitStmt(op->body);
  });

  if (condition_new.same_as(op->condition) && body_new.same_as(op->body)) {
    return ffi::GetRef<Stmt>(op);
  }
  return While(condition_new, body_new, op->span);
}

namespace transform {

/*!
 * \brief The function which returns the pass for the Common Subexpression Elimination.
 * \return The pass for performing CSE.
 */
Pass CommonSubexprElimTIR(bool enable_cse_tir, bool identify_equiv_terms) {
  auto pass_func = [enable_cse_tir, identify_equiv_terms](PrimFunc f, IRModule m, PassContext ctx) {
    if (enable_cse_tir) {
      auto* n = f.CopyOnWrite();
      Context context_init;
      // Add to the initial context all the parameters of the function, as that is needed for
      // doing commoning on terms that use these parameters (it is only possible to introduce
      // a term into a new variable at a specific point in the program if all the variables that
      // it uses have already been declared at this point)
      for (auto current_param : f->params) {
        // The parameters of the functions are variables associated with no value
        context_init.push_back({current_param, MaybeValue()});
      }

      // Do the Common Subexpression Elimination on the body of the function, with the initial
      // context that we have prepared
      n->body = CommonSubexpressionEliminator::PerformCSE(std::move(f->body), context_init,
                                                          identify_equiv_terms);
    }

    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CommonSubexprElimTIR", {});
}

// The pass can now be invoked via the pass infrastructure, but we also add a Python binding for it
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.CommonSubexprElimTIR", CommonSubexprElimTIR);
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm
