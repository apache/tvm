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
 * \file simplify.cc
 * \brief Statement simplifier based on analyzer
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include <optional>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../tir/analysis/control_flow_graph.h"

namespace tvm {
namespace arith {

using namespace tir;

struct SimplifyConfigNode : public tvm::AttrsNode<SimplifyConfigNode> {
  bool transitively_prove_inequalities;
  bool propagate_knowns_to_prove_conditional;
  bool propagate_knowns_to_simplify_expressions;
  bool convert_boolean_to_and_of_ors;
  bool apply_constraints_to_boolean_branches;

  TVM_DECLARE_ATTRS(SimplifyConfigNode, "tir.transform.SimplifyConfig") {
    TVM_ATTR_FIELD(transitively_prove_inequalities)
        .describe(
            "If true, simplify conditionals with transitive combinations of scoped constraints")
        .set_default(false);

    TVM_ATTR_FIELD(propagate_knowns_to_prove_conditional)
        .describe(
            "If true, known buffer values are propagated and used to statically prove conditionals")
        .set_default(false);

    TVM_ATTR_FIELD(propagate_knowns_to_simplify_expressions)
        .describe(
            "If true, known buffer values are propagated and used to replace BufferLoad wherever "
            "possible")
        .set_default(false);

    TVM_ATTR_FIELD(convert_boolean_to_and_of_ors)
        .describe("If true, simplify conditionals into an AND of ORs")
        .set_default(false);

    TVM_ATTR_FIELD(apply_constraints_to_boolean_branches)
        .describe(
            "If true, simplify each branch of AND/OR "
            "under a constraints provided by the other branch")
        .set_default(false);
  }

  RewriteSimplifier::Extension GetEnabledExtensions() const {
    RewriteSimplifier::Extension flags = RewriteSimplifier::kNone;
    if (transitively_prove_inequalities) {
      flags =
          RewriteSimplifier::Extension(flags | RewriteSimplifier::kTransitivelyProveInequalities);
    }
    if (convert_boolean_to_and_of_ors) {
      flags = RewriteSimplifier::Extension(flags | RewriteSimplifier::kConvertBooleanToAndOfOrs);
    }
    if (apply_constraints_to_boolean_branches) {
      flags = RewriteSimplifier::Extension(flags |
                                           RewriteSimplifier::kApplyConstraintsToBooleanBranches);
    }
    return flags;
  }
};

class SimplifyConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SimplifyConfig, Attrs, SimplifyConfigNode);
};

TVM_REGISTER_NODE_TYPE(SimplifyConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.Simplify", SimplifyConfig);

class StmtSimplifier : public IRMutatorWithAnalyzer {
 public:
  static Stmt Apply(Stmt stmt, Analyzer* analyzer, Optional<SimplifyConfig> config_opt = NullOpt) {
    auto config = config_opt.value_or(AttrsWithDefaultValues<arith::SimplifyConfig>());
    analyzer->rewrite_simplify.SetEnabledExtensions(config->GetEnabledExtensions());

    std::optional<ControlFlowGraph> touch_pattern = std::nullopt;
    if (config->propagate_knowns_to_prove_conditional ||
        config->propagate_knowns_to_simplify_expressions) {
      touch_pattern = ControlFlowGraph(stmt);
    }
    StmtSimplifier simplifier(analyzer, config, std::move(touch_pattern));
    return simplifier(std::move(stmt));
  }

 private:
  explicit StmtSimplifier(Analyzer* analyzer, SimplifyConfig config,
                          std::optional<ControlFlowGraph> touch_pattern)
      : IRMutatorWithAnalyzer(analyzer), config_(config), touch_pattern_(touch_pattern) {}

  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  PrimExpr VisitExpr(const PrimExpr& expr) final {
    if (config_->propagate_knowns_to_simplify_expressions) {
      return touch_pattern_->SimplifyInContext(expr, current_stmt_.value(), analyzer_);
    } else {
      return analyzer_->Simplify(expr);
    }
  }

  Stmt Simplify(Stmt stmt) { return operator()(std::move(stmt)); }

  Stmt VisitStmt(const Stmt& stmt) override {
    Optional<Stmt> cache = this->current_stmt_;
    this->current_stmt_ = stmt;
    Stmt output = Parent::VisitStmt(stmt);
    this->current_stmt_ = std::move(cache);
    return output;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
    With<ConstraintContext> ctx2(analyzer_, op->loop_var < op->min + op->extent);
    return Parent::VisitStmt_(op);
  }

  bool CanInlineLetStmt(const LetStmtNode* op) {
    if (is_const_number(op->value)) return true;
    if (op->value.as<VarNode>()) return true;
    // Won't face the deep expression explosion problem as in Let expression.
    // attempt to inline as much as possible if the value integer type(can be index).
    if (!op->value.dtype().is_int()) return false;
    return SideEffect(op->value) <= CallEffectKind::kPure;
  }

  Stmt VisitStmt_(const LetStmtNode* op) override {
    PrimExpr value = this->VisitExpr(op->value);
    if (CanInlineLetStmt(op)) {
      // it is fine to discard the let binding
      // because the call to simplify will always inline the var.
      analyzer_->Bind(op->var, value);
      return this->VisitStmt(op->body);
    } else if (SideEffect(op->value) <= CallEffectKind::kPure) {
      // Even if we aren't replacing all occurrences, they may be
      // necessary for proving conditional statements.
      non_inlined_bindings_.Set(op->var, value);
    }
    Stmt body = this->VisitStmt(op->body);
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->value = std::move(value);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const IfThenElseNode* op) override {
    if (Optional<Bool> cond = ProveCondition(op->condition)) {
      if (cond.value()->value) {
        return this->VisitStmt(op->then_case);
      } else if (op->else_case) {
        return this->VisitStmt(op->else_case.value());
      } else {
        return Evaluate(0);
      }
    } else {
      return Parent::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::if_then_else())) {
      if (Optional<Bool> cond = ProveCondition(op->args[0])) {
        if (cond.value()->value) {
          return this->VisitExpr(op->args[1]);
        } else {
          return this->VisitExpr(op->args[2]);
        }
      }
    }
    return Parent::VisitExpr_(op);
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
  }

  // eliminate useless stores
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(Parent::VisitStmt_(op));
    if (const BufferLoadNode* load = store->value.as<BufferLoadNode>()) {
      if (load->buffer->data.same_as(store->buffer->data) &&
          ArrayDeepEqual(load->indices, store->indices) &&
          tir::ExprDeepEqual()(load->buffer->elem_offset, store->buffer->elem_offset) &&
          ArrayDeepEqual(load->buffer->shape, store->buffer->shape) &&
          ArrayDeepEqual(load->buffer->strides, store->buffer->strides)) {
        return Evaluate(0);
      }
    }
    return std::move(store);
  }

 private:
  bool ArrayDeepEqual(const Array<PrimExpr>& lhs, const Array<PrimExpr>& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
      if (!tir::ExprDeepEqual()(lhs[i], rhs[i])) {
        return false;
      }
    }
    return true;
  }

  /* \brief Internal utility for checking conditionals
   *
   * Uses more aggressive optimization, such as performing additional
   * inlining and tracking known buffer values.
   */
  Optional<Bool> ProveCondition(PrimExpr condition) const {
    condition = Substitute(condition, non_inlined_bindings_);
    if (config_->propagate_knowns_to_prove_conditional) {
      ICHECK(touch_pattern_.has_value());
      condition = touch_pattern_->SimplifyInContext(condition, current_stmt_.value(), analyzer_);
    } else {
      condition = analyzer_->Simplify(condition);
    }
    if (const int64_t* as_int = as_const_int(condition)) {
      return Bool(*as_int);
    } else {
      return NullOpt;
    }
  }

  SimplifyConfig config_;
  std::optional<ControlFlowGraph> touch_pattern_;

  Map<Var, PrimExpr> non_inlined_bindings_;
  Optional<Stmt> current_stmt_{NullOpt};
};

}  // namespace arith

namespace tir {

Stmt Simplify(Stmt stmt, arith::Analyzer* analyzer) {
  return arith::StmtSimplifier::Apply(stmt, analyzer);
}

namespace transform {

Pass Simplify() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    arith::Analyzer analyzer;
    auto cfg = ctx->GetConfig<arith::SimplifyConfig>("tir.Simplify");

    auto* n = f.CopyOnWrite();
    n->body = arith::StmtSimplifier::Apply(std::move(n->body), &analyzer, cfg);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.Simplify", {});
}

TVM_REGISTER_GLOBAL("tir.transform.Simplify").set_body_typed(Simplify);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
