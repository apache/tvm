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

#include "../../tir/transform/simplify.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
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

struct SimplifyConfigNode : public AttrsNodeReflAdapter<SimplifyConfigNode> {
  bool transitively_prove_inequalities;
  bool propagate_knowns_to_prove_conditional;
  bool propagate_knowns_to_simplify_expressions;
  bool convert_boolean_to_and_of_ors;
  bool apply_constraints_to_boolean_branches;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SimplifyConfigNode>()
        .def_ro("transitively_prove_inequalities",
                &SimplifyConfigNode::transitively_prove_inequalities,
                "If true, simplify conditionals with transitive combinations of scoped constraints",
                refl::DefaultValue(false))
        .def_ro(
            "propagate_knowns_to_prove_conditional",
            &SimplifyConfigNode::propagate_knowns_to_prove_conditional,
            "If true, known buffer values are propagated and used to statically prove conditionals",
            refl::DefaultValue(false))
        .def_ro(
            "propagate_knowns_to_simplify_expressions",
            &SimplifyConfigNode::propagate_knowns_to_simplify_expressions,
            "If true, known buffer values are propagated and used to replace BufferLoad wherever "
            "possible",
            refl::DefaultValue(false))
        .def_ro("convert_boolean_to_and_of_ors", &SimplifyConfigNode::convert_boolean_to_and_of_ors,
                "If true, simplify conditionals into an AND of ORs", refl::DefaultValue(false))
        .def_ro("apply_constraints_to_boolean_branches",
                &SimplifyConfigNode::apply_constraints_to_boolean_branches,
                "If true, simplify each branch of AND/OR under a constraints provided by the other "
                "branch",
                refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.transform.SimplifyConfig", SimplifyConfigNode,
                                    BaseAttrsNode);

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
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SimplifyConfig, Attrs, SimplifyConfigNode);
};

TVM_FFI_STATIC_INIT_BLOCK() { SimplifyConfigNode::RegisterReflection(); }

TVM_REGISTER_PASS_CONFIG_OPTION("tir.Simplify", SimplifyConfig);

class StmtSimplifier : public IRMutatorWithAnalyzer {
 public:
  static PrimFunc Apply(PrimFunc func, Analyzer* analyzer,
                        ffi::Optional<SimplifyConfig> config_opt = std::nullopt) {
    auto config = config_opt.value_or(AttrsWithDefaultValues<arith::SimplifyConfig>());
    analyzer->rewrite_simplify.SetEnabledExtensions(config->GetEnabledExtensions());

    std::optional<ControlFlowGraph> touch_pattern = std::nullopt;
    if (config->propagate_knowns_to_prove_conditional ||
        config->propagate_knowns_to_simplify_expressions) {
      touch_pattern = ControlFlowGraph(func->body);
    }

    StmtSimplifier simplifier(analyzer, config, std::move(touch_pattern));
    simplifier.MarkBufferMapShapes(func);
    func.CopyOnWrite()->body = simplifier(func->body);
    return func;
  }

 private:
  explicit StmtSimplifier(Analyzer* analyzer, SimplifyConfig config,
                          std::optional<ControlFlowGraph> touch_pattern)
      : IRMutatorWithAnalyzer(analyzer), config_(config), touch_pattern_(touch_pattern) {}

  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  // Do not simplify buffer definition fields (shape, strides, elem_offset).
  //
  // The simplifier's VisitExpr override calls analyzer_->Simplify() directly,
  // bypassing the normal ExprMutator dispatch. This means BufferLoad expressions
  // inside values (e.g., BufferStore value) skip VisitExpr_(BufferLoadNode*) and
  // thus skip VisitBufferUse. If VisitBufferDef remaps buffers at DeclBuffer sites,
  // the BufferLoad use sites won't pick up the remap, causing DeclBuffer/BufferLoad
  // buffer identity divergence and well-formedness violations.
  //
  // Instead, we keep buffer definitions unchanged and rely on used_in_buffer_def_
  // to prevent inlining LetStmt vars that appear in buffer definitions.
  Buffer VisitBufferDef(const Buffer& buffer, bool alloc_data) override { return buffer; }

  PrimExpr VisitExpr(const PrimExpr& expr) final {
    if (config_->propagate_knowns_to_simplify_expressions) {
      return touch_pattern_->SimplifyInContext(expr, current_stmt_.value(), analyzer_);
    } else {
      return analyzer_->Simplify(expr);
    }
  }

  Stmt Simplify(Stmt stmt) { return operator()(std::move(stmt)); }

  Stmt VisitStmt(const Stmt& stmt) override {
    ffi::Optional<Stmt> cache = this->current_stmt_;
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

  Stmt VisitStmt_(const BindNode* op) override {
    PrimExpr value = this->VisitExpr(op->value);
    // Bind in analyzer for constraint proving and simplification of
    // subsequent expressions.  Don't remove the Bind statement --
    // with flat Bind there's no body to inspect for usage patterns,
    // so we always keep the Bind.
    if (SideEffect(value) <= CallEffectKind::kPure) {
      analyzer_->Bind(op->var, value);
      non_inlined_bindings_.Set(op->var, value);
    }

    if (value.same_as(op->value)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->value = std::move(value);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const IfThenElseNode* op) override {
    if (ffi::Optional<Bool> cond = ProveCondition(op->condition)) {
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
      if (ffi::Optional<Bool> cond = ProveCondition(op->args[0])) {
        if (cond.value()->value) {
          return this->VisitExpr(op->args[1]);
        } else {
          return this->VisitExpr(op->args[2]);
        }
      }
    }
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override { return Parent::VisitExpr_(op); }

  // eliminate useless stores
  Stmt VisitStmt_(const BufferStoreNode* op) override {
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
    return store;
  }

 private:
  bool ArrayDeepEqual(const ffi::Array<PrimExpr>& lhs, const ffi::Array<PrimExpr>& rhs) {
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
  ffi::Optional<Bool> ProveCondition(PrimExpr condition) const {
    condition = Substitute(condition, non_inlined_bindings_);
    if (config_->propagate_knowns_to_prove_conditional) {
      TVM_FFI_ICHECK(touch_pattern_.has_value());
      condition = touch_pattern_->SimplifyInContext(condition, current_stmt_.value(), analyzer_);
    } else {
      condition = analyzer_->Simplify(condition);
    }
    if (const int64_t* as_int = as_const_int(condition)) {
      return Bool(*as_int);
    } else {
      return std::nullopt;
    }
  }

  SimplifyConfig config_;
  std::optional<ControlFlowGraph> touch_pattern_;

  ffi::Map<Var, PrimExpr> non_inlined_bindings_;
  ffi::Optional<Stmt> current_stmt_{std::nullopt};
};

}  // namespace arith

namespace tir {

PrimFunc Simplify(PrimFunc func, arith::Analyzer* analyzer) {
  return arith::StmtSimplifier::Apply(std::move(func), analyzer);
}

namespace transform {

Pass Simplify() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    arith::Analyzer analyzer;
    auto cfg = ctx->GetConfig<arith::SimplifyConfig>("tir.Simplify");

    return arith::StmtSimplifier::Apply(f, &analyzer, cfg);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.Simplify", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.Simplify", Simplify);
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm
