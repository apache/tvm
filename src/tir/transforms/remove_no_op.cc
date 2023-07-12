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
 * \file remove_no_op.cc
 * \brief Remove no op from the stmt
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>
#include <unordered_map>

#include "../../arith/const_fold.h"
#include "../../arith/ir_mutator_with_analyzer.h"
#include "../analysis/control_flow_graph.h"
#include "../analysis/var_use_def_analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

struct RemoveNoOpConfigNode : public tvm::AttrsNode<RemoveNoOpConfigNode> {
  bool use_dataflow_analysis;
  int64_t max_simplification_steps;

  TVM_DECLARE_ATTRS(RemoveNoOpConfigNode, "tir.transform.RemoveNoOpConfig") {
    TVM_ATTR_FIELD(use_dataflow_analysis)
        .describe(
            "If true, known buffer values are propagated and used "
            "to statically prove statements as no-ops.")
        .set_default(false);
    TVM_ATTR_FIELD(max_simplification_steps)
        .describe(
            "If non-zero, RewriteSimplifier will throw an error "
            "after the number of steps specified.  "
            "For use in debug and testing purposes.")
        .set_default(0);
  }
};

class RemoveNoOpConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RemoveNoOpConfig, Attrs, RemoveNoOpConfigNode);
};

TVM_REGISTER_NODE_TYPE(RemoveNoOpConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.RemoveNoOp", RemoveNoOpConfig);

// Mark the statement of each stage.
class NoOpRemover : public arith::IRMutatorWithAnalyzer {
 public:
  static Stmt Apply(Stmt stmt, arith::Analyzer* analyzer,
                    std::optional<ControlFlowGraph> touch_pattern, const StmtNode* context) {
    NoOpRemover visitor(analyzer, touch_pattern, context);
    return visitor(std::move(stmt));
  }

 private:
  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  NoOpRemover(arith::Analyzer* analyzer, std::optional<ControlFlowGraph> touch_pattern,
              const StmtNode* context)
      : Parent(analyzer), touch_pattern_(touch_pattern), context_(context) {}

  Stmt VisitStmt_(const LetStmtNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<LetStmtNode>();
    if (is_no_op(op->body)) {
      return MakeEvaluate(op->value);
    }

    bool body_uses_bound_variable =
        !UsesVar(op->body, [&](const VarNode* var) { return var == op->var.get(); });
    if (body_uses_bound_variable && HasSideEffect(op->value)) {
      return SeqStmt({MakeEvaluate(op->value), op->body});
    } else if (body_uses_bound_variable) {
      return op->body;
    } else {
      return stmt;
    }
  }
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == "pragma_debug_skip_region") {
      return MakeEvaluate(0);
    } else if (op->attr_key == attr::async_wait_queue_scope) {
      auto wait_attrs = GetAsyncWaitAttributes(op);
      auto wait_cnt = wait_attrs.second;
      arith::Analyzer ana;
      if (ana.CanProve(wait_cnt < 0)) {
        // A negative wait count can arise if it depends on a loop variable.
        // For example, a wait count 1 - i can be negative after loop unrolling.
        // We assume that such wait is a nop.
        auto inner = op->body.as<AttrStmtNode>();
        ICHECK(inner);
        return Parent::VisitStmt(inner->body);
      }
    }

    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<AttrStmtNode>();
    return is_no_op(op->body) ? MakeEvaluate(op->value) : stmt;
  }
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<IfThenElseNode>();
    // Sometimes the condition can be statically determined,
    // in which the type of the `stmt` will not be IfThenElseNode.
    if (!op) {
      return stmt;
    }
    if (op->else_case) {
      bool no_op_else = is_no_op(op->else_case.value());
      bool no_op_then = is_no_op(op->then_case);
      if (no_op_else && no_op_then) {
        return MakeEvaluate(op->condition);
      } else if (no_op_else) {
        return IfThenElse(op->condition, op->then_case);
      } else if (no_op_then) {
        return IfThenElse(!op->condition, op->else_case.value());
      } else {
        return stmt;
      }
    } else {
      if (is_no_op(op->then_case)) {
        return MakeEvaluate(op->condition);
      } else {
        return stmt;
      }
    }
  }
  Stmt VisitStmt_(const ForNode* op) final {
    auto extent_range = arith::EvalSet(op->extent, var_range_map_);
    if (!arith::is_neg_inf(extent_range.max()) && !arith::is_pos_inf(extent_range.max()) &&
        analyzer_->CanProve(extent_range.max() <= 0)) {
      return Evaluate(0);
    }
    var_range_map_[op->loop_var.get()] = arith::IntSet::FromMinExtent(op->min, op->extent);
    Stmt stmt = Parent::VisitStmt_(op);
    var_range_map_.erase(op->loop_var.get());
    op = stmt.as<ForNode>();
    if (is_zero(op->extent)) {
      return Evaluate(0);
    }
    return is_no_op(op->body) ? MakeEvaluate({op->min, op->extent}) : stmt;
  }
  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    return is_no_op(op->body) ? MakeEvaluate(op->extents) : stmt;
  }

  Stmt VisitStmt_(const ProducerRealizeNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<ProducerRealizeNode>();
    return is_no_op(op->body) ? op->body : stmt;
  }
  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (HasSideEffect(op->value)) {
      return GetRef<Stmt>(op);
    } else {
      return Evaluate(0);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = GetRef<BufferStore>(op);

    // Helper function that returns a statement containing only the
    // side effects of evaluating this BufferStore, but not the store
    // itself.
    auto only_side_effects = [&]() {
      Array<Stmt> statements;
      statements.push_back(MakeEvaluate(store->value));
      for (const auto& index : store->indices) {
        statements.push_back(MakeEvaluate(index));
      }
      return this->VisitStmt(SeqStmt(statements));
    };

    if (touch_pattern_.has_value()) {
      // A write that is later overwritten is a no-op.
      Stmt context = context_ ? GetRef<Stmt>(context_) : store;
      if (touch_pattern_->IsOverwrittenWithoutEffect(store, context)) {
        touch_pattern_->RemoveStore(store);
        return only_side_effects();
      }
    }

    // A write whose destination is known to already contain the
    // values to be written is a no-op.
    // PrimExpr stores_existing_value = store->value == BufferLoad(store->buffer, store->indices);
    PrimExpr stores_existing_value = store->value - BufferLoad(store->buffer, store->indices) == 0;
    if (touch_pattern_.has_value()) {
      Stmt context_arg = context_ ? GetRef<Stmt>(context_) : Stmt(store);
      stores_existing_value =
          touch_pattern_->SimplifyInContext(stores_existing_value, context_arg, analyzer_);
    } else {
      stores_existing_value = analyzer_->Simplify(stores_existing_value);
    }
    if (is_one(stores_existing_value)) {
      return only_side_effects();
    }

    // If the stored value is a load from the same location, the
    // statement is a no-op, regardless of contextual information.
    if (const BufferLoadNode* load = store->value.as<BufferLoadNode>()) {
      if (load->buffer->data.same_as(store->buffer->data) &&
          analyzer_->CanProveEqual(load->buffer->elem_offset, store->buffer->elem_offset) &&
          ArrayValueEqual(load->buffer->shape, store->buffer->shape) &&
          ArrayValueEqual(load->buffer->strides, store->buffer->strides) &&
          ArrayValueEqual(load->indices, store->indices)) {
        return only_side_effects();
      }
    }

    return std::move(store);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    auto node = Downcast<DeclBuffer>(Parent::VisitStmt_(op));

    VarUseDefAnalyzer var_use({});
    var_use(node->body);

    if (var_use.buffer_use_count_.count(node->buffer.get())) {
      return std::move(node);
    } else {
      return node->body;
    }
  }

 private:
  bool ArrayValueEqual(const Array<PrimExpr>& a, const Array<PrimExpr>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
      if (!analyzer_->CanProveEqual(a[i], b[i])) {
        return false;
      }
    }
    return true;
  }

  bool HasSideEffect(const PrimExpr& value) {
    return SideEffect(value) > CallEffectKind::kReadState;
  }

  Stmt MakeEvaluate(PrimExpr value) {
    if (SideEffect(value) > CallEffectKind::kReadState) {
      return Evaluate(value);
    } else {
      return Evaluate(0);
    }
  }
  Stmt MakeEvaluate(const Array<PrimExpr>& values) {
    Array<Stmt> stmts;
    for (PrimExpr e : values) {
      if (SideEffect(e) > CallEffectKind::kReadState) {
        stmts.push_back(Evaluate(e));
      }
    }

    if (stmts.size() == 0) {
      return Evaluate(0);
    } else if (stmts.size() == 1) {
      return stmts[0];
    } else {
      return SeqStmt(stmts);
    }
  }

  std::unordered_map<const VarNode*, arith::IntSet> var_range_map_;
  std::optional<ControlFlowGraph> touch_pattern_;
  const StmtNode* context_;
};

Stmt RemoveNoOp(Stmt stmt, arith::Analyzer* analyzer, std::optional<ControlFlowGraph> touch_pattern,
                const StmtNode* context) {
  return NoOpRemover::Apply(std::move(stmt), analyzer, std::move(touch_pattern), context);
}

namespace transform {

Pass RemoveNoOp() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    std::optional<ControlFlowGraph> touch_pattern = std::nullopt;

    RemoveNoOpConfig config = ctx->GetConfig<RemoveNoOpConfig>("tir.RemoveNoOp")
                                  .value_or(AttrsWithDefaultValues<RemoveNoOpConfig>());

    if (config->use_dataflow_analysis) {
      touch_pattern.emplace(f->body, config->max_simplification_steps);
    }

    arith::Analyzer analyzer;
    analyzer.rewrite_simplify.SetMaximumRewriteSteps(config->max_simplification_steps);

    {
      auto* write_ptr = f.CopyOnWrite();
      write_ptr->body = NoOpRemover::Apply(std::move(write_ptr->body), &analyzer,
                                           std::move(touch_pattern), nullptr);
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveNoOp", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveNoOp").set_body_typed(RemoveNoOp);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
