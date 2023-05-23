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
 * \file reduce_branching_through_overcompute.cc
 *
 * \brief Attempt to remove conditional statements by introducing
 * extra computations that do not impact the final results.
 */

#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include <optional>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../analysis/control_flow_graph.h"
#include "remove_no_op.h"
#include "simplify.h"

namespace tvm {
namespace tir {

struct ReduceBranchingThroughOvercomputeConfigNode
    : public tvm::AttrsNode<ReduceBranchingThroughOvercomputeConfigNode> {
  bool use_dataflow_analysis;

  TVM_DECLARE_ATTRS(ReduceBranchingThroughOvercomputeConfigNode,
                    "tir.transform.ReduceBranchingThroughOvercomputeConfig") {
    TVM_ATTR_FIELD(use_dataflow_analysis)
        .describe(
            "If true, known buffer values are propagated and used "
            "to statically prove that overcompute is valid.")
        .set_default(false);
  }
};

class ReduceBranchingThroughOvercomputeConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ReduceBranchingThroughOvercomputeConfig, Attrs,
                                            ReduceBranchingThroughOvercomputeConfigNode);
};

TVM_REGISTER_NODE_TYPE(ReduceBranchingThroughOvercomputeConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.ReduceBranchingThroughOvercompute",
                                ReduceBranchingThroughOvercomputeConfig);

struct ElseBranchFiller : StmtExprMutator {
  Stmt VisitStmt_(const IfThenElseNode* op) override {
    IfThenElse ret = Downcast<IfThenElse>(StmtExprMutator::VisitStmt_(op));
    if (ret->else_case.defined()) {
      return std::move(ret);
    } else {
      auto new_else_clause = Evaluate(0);
      new_else_clauses.insert(new_else_clause);
      return IfThenElse(ret->condition, ret->then_case, new_else_clause);
    }
  }

  std::unordered_set<Evaluate, ObjectPtrHash, ObjectPtrEqual> new_else_clauses;
};

class ElseBranchStripper : public StmtExprMutator {
 public:
  ElseBranchStripper(
      const std::unordered_set<Evaluate, ObjectPtrHash, ObjectPtrEqual>& new_else_clauses)
      : new_else_clauses_(new_else_clauses) {}

 private:
  Stmt VisitStmt_(const IfThenElseNode* op) override {
    IfThenElse ret = Downcast<IfThenElse>(StmtExprMutator::VisitStmt_(op));
    if (auto as_eval = ret->else_case.as<Evaluate>();
        as_eval && new_else_clauses_.count(as_eval.value())) {
      return IfThenElse(ret->condition, ret->then_case);
    } else {
      return std::move(ret);
    }
  }

  const std::unordered_set<Evaluate, ObjectPtrHash, ObjectPtrEqual>& new_else_clauses_;
};

class BranchReducer : public arith::IRMutatorWithAnalyzer {
 public:
  static Stmt Apply(Stmt stmt, const std::optional<ControlFlowGraph>& touch_pattern) {
    arith::Analyzer analyzer;
    BranchReducer visitor(&analyzer, touch_pattern);
    return visitor(std::move(stmt));
  }

 private:
  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  BranchReducer(arith::Analyzer* analyzer, const std::optional<ControlFlowGraph>& touch_pattern)
      : Parent(analyzer), touch_pattern_(touch_pattern) {}

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    IfThenElse cond = Downcast<IfThenElse>(Parent::VisitStmt_(op));

    auto is_special_case = [&](PrimExpr condition, Stmt general_case, Stmt special_case) -> bool {
      condition = analyzer_->rewrite_simplify(condition);
      With<arith::ConstraintContext> constraint(analyzer_, condition);
      Stmt stmt = RemoveNoOp(general_case, analyzer_, touch_pattern_, special_case.get());
      return StructuralEqual()(stmt, special_case);
    };

    ICHECK(cond->else_case.defined() || !touch_pattern_.has_value())
        << "Temp assert, should be true whenever touch pattern is available";
    Stmt else_case = cond->else_case.value_or(Evaluate(0));

    if (is_special_case(cond->condition, else_case, cond->then_case)) {
      return else_case;
    } else if (is_special_case(!cond->condition, cond->then_case, else_case)) {
      return cond->then_case;
    } else {
      return std::move(cond);
    }
  }

 private:
  const std::optional<ControlFlowGraph>& touch_pattern_;
};

namespace transform {

Pass ReduceBranchingThroughOvercompute() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    arith::Analyzer analyzer;

    ReduceBranchingThroughOvercomputeConfig config =
        ctx->GetConfig<ReduceBranchingThroughOvercomputeConfig>(
               "tir.ReduceBranchingThroughOvercompute")
            .value_or(AttrsWithDefaultValues<ReduceBranchingThroughOvercomputeConfig>());

    auto* n = f.CopyOnWrite();

    std::optional<ControlFlowGraph> touch_pattern = std::nullopt;
    ElseBranchFiller else_branch_filler;
    if (config->use_dataflow_analysis) {
      n->body = else_branch_filler(std::move(n->body));
      touch_pattern.emplace(n->body);
    }

    n->body = BranchReducer::Apply(std::move(n->body), touch_pattern);

    if (config->use_dataflow_analysis) {
      n->body = ElseBranchStripper(else_branch_filler.new_else_clauses)(std::move(n->body));
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ReduceBranchingThroughOvercompute", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ReduceBranchingThroughOvercompute")
    .set_body_typed(ReduceBranchingThroughOvercompute);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
