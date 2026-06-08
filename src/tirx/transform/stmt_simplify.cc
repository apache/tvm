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
 * \file stmt_simplify.cc
 * \brief Statement simplifier based on analyzer
 */

#include "../../tirx/transform/stmt_simplify.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace arith {

using namespace tirx;

struct StmtSimplifyConfigNode : public ffi::Object {
  bool transitively_prove_inequalities;
  bool convert_boolean_to_and_of_ors;
  bool apply_constraints_to_boolean_branches;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StmtSimplifyConfigNode>()
        .def_ro("transitively_prove_inequalities",
                &StmtSimplifyConfigNode::transitively_prove_inequalities,
                "If true, simplify conditionals with transitive combinations of scoped constraints",
                refl::DefaultValue(false))
        .def_ro("convert_boolean_to_and_of_ors",
                &StmtSimplifyConfigNode::convert_boolean_to_and_of_ors,
                "If true, simplify conditionals into an AND of ORs", refl::DefaultValue(false))
        .def_ro("apply_constraints_to_boolean_branches",
                &StmtSimplifyConfigNode::apply_constraints_to_boolean_branches,
                "If true, simplify each branch of AND/OR under constraints provided by the other "
                "branch",
                refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.transform.StmtSimplifyConfig", StmtSimplifyConfigNode,
                                    ffi::Object);

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

class StmtSimplifyConfig : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtSimplifyConfig, ffi::ObjectRef,
                                                StmtSimplifyConfigNode);
};

static StmtSimplifyConfig MakeDefaultStmtSimplifyConfig() {
  return tvm::transform::PassConfigWithDefaults<StmtSimplifyConfig>();
}

TVM_FFI_STATIC_INIT_BLOCK() { StmtSimplifyConfigNode::RegisterReflection(); }

TVM_REGISTER_PASS_CONFIG_OPTION("tirx.StmtSimplify", StmtSimplifyConfig);

class StmtSimplifier : public IRMutatorWithAnalyzer {
 public:
  static PrimFunc Apply(PrimFunc func, AnalyzerObj* analyzer,
                        ffi::Optional<StmtSimplifyConfig> config_opt = std::nullopt) {
    auto config = config_opt.value_or(MakeDefaultStmtSimplifyConfig());
    analyzer->rewrite_simplify.SetEnabledExtensions(config->GetEnabledExtensions());

    StmtSimplifier simplifier(analyzer, config);
    simplifier.MarkBufferMapShapes(func);
    func.CopyOnWrite()->body = simplifier(func->body);
    return func;
  }

 private:
  explicit StmtSimplifier(AnalyzerObj* analyzer, StmtSimplifyConfig config)
      : IRMutatorWithAnalyzer(analyzer), config_(config) {}

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

  PrimExpr VisitExpr(const PrimExpr& expr) final { return analyzer_->Simplify(expr); }

  Stmt Simplify(Stmt stmt) { return operator()(std::move(stmt)); }

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
      // Record the binding so we can substitute it into assert conditions
      // (see VisitStmt_(const AssertStmtNode*)).  Under SSA each var is
      // bound exactly once, so the map grows monotonically without key
      // conflicts.  No scope-based cleanup is needed because vars bound
      // in inner scopes are only referenced within those scopes; stale
      // entries are harmless and never consulted again.
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
    if (ffi::Optional<bool> cond = ProveCondition(op->condition)) {
      if (cond.value()) {
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
      if (ffi::Optional<bool> cond = ProveCondition(op->args[0])) {
        if (cond.value()) {
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
          tirx::ExprDeepEqual()(load->buffer->elem_offset, store->buffer->elem_offset) &&
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
      if (!tirx::ExprDeepEqual()(lhs[i], rhs[i])) {
        return false;
      }
    }
    return true;
  }

  /* \brief Internal utility for checking conditionals
   *
   * Substitutes any known Bind values and then simplifies with the analyzer.
   */
  ffi::Optional<bool> ProveCondition(PrimExpr condition) const {
    condition = Substitute(condition, non_inlined_bindings_);
    condition = analyzer_->Simplify(condition);
    if (const int64_t* as_int = as_const_int(condition)) {
      return *as_int != 0;
    } else {
      return std::nullopt;
    }
  }

  StmtSimplifyConfig config_;

  // Pure Bind values kept for substitution into assert conditions.
  // Grows monotonically under SSA — no scope-based cleanup required.
  ffi::Map<Var, PrimExpr> non_inlined_bindings_;
};

}  // namespace arith

namespace tirx {

PrimFunc StmtSimplify(PrimFunc func, arith::AnalyzerObj* analyzer) {
  return arith::StmtSimplifier::Apply(std::move(func), analyzer);
}

namespace transform {

Pass StmtSimplify() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    arith::Analyzer analyzer;
    auto cfg = ctx->GetConfig<arith::StmtSimplifyConfig>("tirx.StmtSimplify");

    return arith::StmtSimplifier::Apply(f, analyzer.get(), cfg);
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.StmtSimplify", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.StmtSimplify", StmtSimplify);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
