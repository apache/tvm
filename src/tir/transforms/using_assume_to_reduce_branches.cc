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
 * \file using_assume_to_reduce_branches.cc
 *
 * \brief Attempt to remove conditional branch statements by introducing
 * extra computations that do not impact the final results. Mainly
 * oriented for layout specific padding related branches.
 *
 * \note
 *    1. This pass works if the buffer assumption variable is in the branch statement.
 *       In case, the buffer assumption is not present in the branch statement and
 *       there are intermediate buffers then, inline the code.
 *    2. The assumptions leveraged here should be of the form T.assume(condition_on_indices or
 *       buffer_equals_to_some_value)
 *    3. Some part of the code are reused from the control_flow_graph.cc file which also
 *       handles eliminating branches in particular scenarios.
 *    4. This pass currently works for op_pattern kElemWise and kBroadcast.
 */

#include <tvm/relax/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>

#include "../../arith/constraint_extract.h"
#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/unwrap_vector_expr.h"
#include "simplify.h"
#include "tvm/ir/expr.h"
namespace tvm {
namespace tir {

using namespace arith;

class AssumeChecker : public StmtExprVisitor {
  /* This class checks if the primfunc has assume statement.
  If yes, then only the FuncAnanlyzerMutator class runs. This is to ensure speedup in the pass.*/
 public:
  bool has_assume = false;

  void VisitStmt(const Stmt& stmt) final {
    if (has_assume) {
      return;
    }
    StmtVisitor::VisitStmt(stmt);
  }
  void VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::assume())) {
      has_assume = true;
    }
  }
};

class ParseAssumeAndOvercompute : public IRMutatorWithAnalyzer {
  /* This class analyzes the complete primfunc.
  It parses the buffer assumptions and eliminates the redundant branch
  introduced due to layout specific padding by leveraging from buffer assumptions.
  On eliminating the branch there are more opportunities to vectorize the code
  and improve performance.

  Example:
  -------------
  Prim Func Before :
  for (...)
    T.assume( assume_condition or A[i] == 0 )
  for (...)
    out = T.if_then_else(if_then_else_condition, 0, function(A))
    # here function(A) is some function on Var A

  Prim Func After :
    for (...)
    T.assume( assume_condition or A[i] == 0 )
  for (...)
    out = function(A) # here function(A) is some function on the Var A
  --------------
  # High-level implementation details :
    1. The pass parses the assume statement and stores the relevant information.
    2. The pass tries to evaluate the then_clause and else_clause in then_condition_context
    and else_condition_context.
    It checks if the context of the assume statement (for condition indices and
    assume_condition) is same as the context of the if_then_else statement (for condition indices
    and if_then_else condition). If context is same and the expression inside if_then_else statement
    is a function of the buffer assumption (eg A in above example),
    then the pass substitutes the value from the buffer assumption and simplifies the expression.
    3. The pass then checks if then_clause and else_clause evaluate to same value.
    If yes, then return the else_clause if we are in the then_condition_context (since then_clause
    will be true in this context and if else_clause is also evaluating to true then we can directly
    replace it with else_clause), similarly, we return the then_clause if we are in the
    else_condition_context.
  This class handles all these scenarios.*/

 public:
  using Parent = IRMutatorWithAnalyzer;
  explicit ParseAssumeAndOvercompute(Analyzer* analyzer) : Parent(analyzer) {}

 private:
  using Parent::VisitExpr_;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  // This struct stores all the relevant data related to asssume statement
  struct assume_struct {             // Consider the example : T.assume(i < 14 or A[i] == 0)
    PrimExpr buffer_context;         // The context of the assume statement (the bound on the axis)
    PrimExpr buffer_predicate;       // The condition inside assume statement (i < 14) excluding
                                     // bufferload expression (A[i] == 0)
    tir::BufferLoad buffer_load;     // Storing the buffer load Eg: A[i] in A[i] == 0
    PrimExpr buffer_value;           // Storing the value for the buffer Eg : 0 in A[i] == 0
    Array<PrimExpr> buffer_indices;  // Storing the indices of the buffer Eg : i
  };
  // List of conditions in a scope
  std::vector<PrimExpr> conditions_;

  // Storing all the buffer assumptions data in map
  std::map<tir::Buffer, assume_struct> map_buffer_assumption;
  tir::Buffer current_bufferstorenode_name;

  struct InternalConstraintContext {
    /* This stuct appends the constraint passed to it in the conditions list.
    It keeps track of the bounds of the variables along with any conditions on the variables */
    InternalConstraintContext(ParseAssumeAndOvercompute* self, PrimExpr constraint)
        : self(self), analyzer_context(self->analyzer_, constraint) {
      old_num_constraints = self->conditions_.size();

      auto side_effect = tir::SideEffect(constraint);
      if (side_effect <= tir::CallEffectKind::kPure) {
        self->conditions_.push_back(constraint);
      } else if (side_effect <= tir::CallEffectKind::kReadState) {
        assume = constraint;
      }

      new_num_constraints = self->conditions_.size();
    }

    ~InternalConstraintContext() {
      ICHECK_EQ(self->conditions_.size(), new_num_constraints)
          << "Internal error: Each condition should only be popped once.";
      self->conditions_.erase(self->conditions_.begin() + old_num_constraints,
                              self->conditions_.end());
    }

    ParseAssumeAndOvercompute* self{nullptr};
    With<arith::ConstraintContext> analyzer_context;
    size_t old_num_constraints{0};
    size_t new_num_constraints{0};
    Optional<PrimExpr> assume{NullOpt};

    // Disable default-generated copy/move assignment and constructors
    InternalConstraintContext(const InternalConstraintContext&) = delete;
    InternalConstraintContext& operator=(const InternalConstraintContext&) = delete;
    InternalConstraintContext(InternalConstraintContext&&) = delete;
    InternalConstraintContext& operator=(InternalConstraintContext&&) = delete;
  };

  PrimExpr CurrentScopePredicate() const {
    /* This combines all the constraints in a scope */
    PrimExpr predicate = Bool(true);
    for (const auto& condition : conditions_) {
      predicate = predicate && condition;
    }
    return predicate;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    /* Create and delete the scope with bind.
    Add the minimum and maximum bound for the variables to the conditions_ list using
    InternalConstraintContext */
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    InternalConstraintContext ctx1(this, op->loop_var >= op->min);
    InternalConstraintContext ctx2(this, op->loop_var < op->min + op->extent);
    return Parent::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    if (map_buffer_assumption.find(op->buffer) != map_buffer_assumption.end()) {
      PrimExpr buf_value;
      /* If the cuurent context where the buffer load is present is same as
      the context of the buffer assumption then, return the buffer value present in the assumption.
      This will eventually replace the bufferload value in the complete expresison */

      auto buffer_assumption = map_buffer_assumption[op->buffer];
      PrimExpr current_predicate_and_context = CurrentScopePredicate();
      PrimExpr buffer_predicate_and_context =
          buffer_assumption.buffer_context && buffer_assumption.buffer_predicate;
      bool current_context_and_buffer_constraint_is_same = StructuralEqual()(
          current_predicate_and_context, buffer_predicate_and_context, /*map_free_vars=*/true);

      if (current_context_and_buffer_constraint_is_same) {
        buf_value = buffer_assumption.buffer_value;
        return buf_value;
      }
    }
    return GetRef<PrimExpr>(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(Parent::VisitStmt_(op));

    // Eliminate the builtin if_then_else statement
    if (auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::if_then_else())) {
        PrimExpr cond = call->args[0];
        PrimExpr then_clause = call->args[1];
        PrimExpr else_clause = call->args[2];

        PrimExpr then_clause_in_then_context;
        PrimExpr else_clause_in_then_context;
        PrimExpr then_clause_in_else_context;
        PrimExpr else_clause_in_else_context;
        {
          // Simplifying expressions in " then context "
          InternalConstraintContext then_ctx(this, cond);
          // This will call the current class's appropriate VisitStmt function
          then_clause_in_then_context = (*this)(then_clause);
          then_clause_in_then_context = analyzer_->Simplify(then_clause_in_then_context);

          else_clause_in_then_context = (*this)(else_clause);
          else_clause_in_then_context = analyzer_->Simplify(else_clause_in_then_context);
        }
        {
          // Simplifying expressions in " else context "
          InternalConstraintContext else_ctx(this, !cond);
          // This will call the current class's appropriate VisitStmt function
          then_clause_in_else_context = (*this)(then_clause);
          then_clause_in_else_context = analyzer_->Simplify(then_clause_in_else_context);

          else_clause_in_else_context = (*this)(else_clause);
          else_clause_in_else_context = analyzer_->Simplify(else_clause_in_else_context);
        }

        auto n = this->CopyOnWrite(op);
        if (StructuralEqual()(then_clause_in_then_context, else_clause_in_then_context)) {
          n->value = analyzer_->Simplify(else_clause);
          return Stmt(n);
        } else if (StructuralEqual()(then_clause_in_else_context, else_clause_in_else_context)) {
          n->value = analyzer_->Simplify(then_clause);
          return Stmt(n);
        } else {
          return Parent::VisitStmt_(op);
        }
      }
    }
    return Parent::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::assume())) {
      Assume(op->args[0]);
    }
    return Parent::VisitExpr_(op);
  }

  void Assume(PrimExpr assumption) {
    for (const auto& expr : arith::ExtractConstraints(assumption, false)) {
      AssumeConstraintComponent(expr);
    }
  }

  void AssumeConstraintComponent(PrimExpr assumption) {
    PrimExpr additional_predicate = Bool(true);
    assume_struct buf_data;

    std::vector<PrimExpr> buffer_exprs;
    for (const auto& expr : arith::ExtractComponents(assumption)) {
      auto side_effect = tir::SideEffect(expr);
      if (side_effect <= tir::CallEffectKind::kPure) {
        // Pulling out portions of the assumption that do not depend
        // on a buffer value allows the following two forms to be
        // treated identically.
        //
        // Option 1: if i < 3: T.assume(buf[i] == value)
        // Option 2: T.assume(i>=3 or buf[i] == value)
        additional_predicate = additional_predicate && logical_not(expr);
      } else if (side_effect == tir::CallEffectKind::kReadState) {
        buffer_exprs.push_back(expr);
      } else {
        LOG(FATAL) << "Assumption must be pure or read-only, but contained expression " << expr
                   << " with side-effect \'" << side_effect << "\'";
      }
    }

    additional_predicate = analyzer_->Simplify(std::move(additional_predicate));
    CHECK_EQ(buffer_exprs.size(), 1) << "T.assume must contain only a single buffer expression";

    auto* as_equal_node = buffer_exprs[0].as<tir::EQNode>();
    CHECK(as_equal_node) << "T.assume buffer constraint must be of the form 'buffer[indices] == "
                            "value', but received "
                         << assumption;
    if (!as_equal_node) {
      // This assumption is an inequality on a data-dependent
      // conditional.  Not an error for this to occur, but also not
      // something that is currently supported.
      return;
    }

    // Parse the statement and store the desired values
    // Ex: A[i]==0, load = A[i], value = 0
    tir::BufferLoad load;
    PrimExpr value;
    if (auto opt = as_equal_node->a.as<tir::BufferLoad>()) {
      load = opt.value();
      value = as_equal_node->b;
    } else if (auto opt = as_equal_node->b.as<tir::BufferLoad>()) {
      load = opt.value();
      value = as_equal_node->a;
    } else {
      LOG(FATAL) << "T.assume buffer constraint must be of the form 'buffer[indices] == value'";
    }

    // Populating the assume statement predicate, buffer, value
    // and the context of the assume statement
    buf_data.buffer_context = CurrentScopePredicate();
    buf_data.buffer_predicate = additional_predicate;
    buf_data.buffer_load = load;
    buf_data.buffer_value = value;
    buf_data.buffer_indices = load->indices;
    for (size_t i = 0; i < load->indices.size(); i++) {
      buf_data.buffer_indices.push_back(analyzer_->Simplify(load->indices[i]));
    }
    map_buffer_assumption[buf_data.buffer_load->buffer] = buf_data;

    auto has_side_effect = tir::SideEffect(value) > tir::CallEffectKind::kPure;
    CHECK(!has_side_effect) << "Buffer value in constraint must be pure expression, but was "
                            << value;
    if (has_side_effect) {
      return;
    }
  }
};

namespace transform {

Pass UseAssumeToReduceBranches() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    arith::Analyzer analyzer;

    // The pass runs & eliminates pad branch with overcompute only if,
    // the primfunc has op_pattern defined and is an elementwise op.
    // AnnotateTIROpPattern pass will set op_pattern in op attributes of the primfunc.
    if (n->attrs.GetAttr<Integer>("op_pattern").defined()) {
      Optional<Integer> opt_pattern = f->GetAttr<Integer>("op_pattern");
      if (opt_pattern.defined()) {
        relay::OpPatternKind pattern;
        pattern = static_cast<relay::OpPatternKind>(Downcast<IntImm>(opt_pattern)->value);

        if (pattern == relay::OpPatternKind::kElemWise ||
            pattern == relay::OpPatternKind::kBroadcast) {
          // If the primfunc contains assume statement then, run the mutator pass.
          AssumeChecker assume_checker;
          assume_checker(std::move(n->body));

          if (assume_checker.has_assume) {
            // Leverage from assume and eliminate the branch
            ParseAssumeAndOvercompute func_analyzer_mutator(&analyzer);
            n->body = func_analyzer_mutator(std::move(n->body));
          }
        }
      }
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.UseAssumeToReduceBranches", {});
}

TVM_REGISTER_GLOBAL("tir.transform.UseAssumeToReduceBranches")
    .set_body_typed(UseAssumeToReduceBranches);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
