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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/transform.h>

#include "../ir_mutator_with_analyzer.h"

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
  using IRMutatorWithAnalyzer::Mutate;
  using IRMutatorWithAnalyzer::Mutate_;
  static PrimFunc Apply(PrimFunc func, const Analyzer& analyzer,
                        ffi::Optional<StmtSimplifyConfig> config_opt = std::nullopt) {
    auto config = config_opt.value_or(MakeDefaultStmtSimplifyConfig());
    analyzer->rewrite_simplify.SetEnabledExtensions(config->GetEnabledExtensions());

    auto simplifier = ffi::make_object<StmtSimplifier>(analyzer, config);
    simplifier->MarkBufferParamShapes(func);
    auto* n = func.CopyOnWrite();
    n->body = simplifier->Mutate(n->body, InplaceMode::kAllow).ValueOrUnchanged(n->body);
    return func;
  }

 public:
  explicit StmtSimplifier(const Analyzer& analyzer, StmtSimplifyConfig config)
      : IRMutatorWithAnalyzer(analyzer), config_(config) {}

 private:
  using Parent = IRMutatorWithAnalyzer;

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input, InplaceMode inplace_mode) final {
    if (input.as<BufferType>()) {
      return ffi::Unchanged();
    }
    if (auto expr = input.as<PrimExpr>()) {
      PrimExpr simplified = analyzer_->Simplify(*expr);
      if (simplified.same_as(*expr)) return ffi::Unchanged();
      return simplified;
    }
    return Parent::Mutate(input, inplace_mode);
  }

  Stmt Simplify(Stmt stmt) { return Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt); }

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
    With<ConstraintContext> ctx2(analyzer_,
                                 static_cast<PrimExpr>(op->loop_var) < op->min + op->extent);
    return Parent::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) override {
    auto prim_value = op->value.as<PrimExpr>();
    if (!prim_value) {
      return Parent::Mutate_(op, inplace_mode);
    }
    PrimExpr value =
        this->Mutate(prim_value.value(), inplace_mode).ValueOrUnchanged(prim_value.value());
    // Bind in analyzer for constraint proving and simplification of
    // subsequent expressions.  Don't remove the Bind statement --
    // with flat Bind there's no body to inspect for usage patterns,
    // so we always keep the Bind.
    if (SideEffect(value) <= CallEffectKind::kPure) {
      analyzer_->Bind(op->var, value);
      // Record the binding so we can substitute it into assert conditions
      // (see Mutate_(const AssertStmtNode*, InplaceMode)).  Under SSA each var is
      // bound exactly once, so the map grows monotonically without key
      // conflicts.  No scope-based cleanup is needed because vars bound
      // in inner scopes are only referenced within those scopes; stale
      // entries are harmless and never consulted again.
      non_inlined_bindings_.Set(op->var, value);
    }

    if (value.same_as(op->value)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* n = const_cast<BindNode*>(op);
        n->value = std::move(value);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<BindNode>(*op);
      n->value = std::move(value);
      return Stmt(n);
    }
  }

  UnchangedOr<Stmt> Mutate_(const IfThenElseNode* op, InplaceMode inplace_mode) override {
    if (ffi::Optional<bool> cond = ProveCondition(op->condition)) {
      if (cond.value()) {
        return this->Mutate(op->then_case, inplace_mode).ValueOrUnchanged(op->then_case);
      } else if (op->else_case) {
        return this->Mutate(op->else_case.value(), inplace_mode)
            .ValueOrUnchanged(op->else_case.value());
      } else {
        return Evaluate(0);
      }
    } else {
      return Parent::Mutate_(op, inplace_mode);
    }
  }

  // eliminate useless stores
  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) override {
    BufferStore store = Parent::Mutate_(op, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                            .as_or_throw<BufferStore>();
    if (const TensorLoadNode* load = store->value.as<TensorLoadNode>()) {
      BufferVar buffer = load->source.as_or_throw<tvm::tirx::BufferVar>();
      if (buffer.same_as(store->buffer) && ArrayDeepEqual(load->indices, store->indices) &&
          prim::ExprDeepEqual()(buffer->elem_offset, store->buffer->elem_offset) &&
          ArrayDeepEqual(buffer->shape, store->buffer->shape) &&
          ArrayDeepEqual(buffer->strides, store->buffer->strides)) {
        return Evaluate(0);
      }
    }
    return store;
  }
  bool ArrayDeepEqual(const ffi::Array<PrimExpr>& lhs, const ffi::Array<PrimExpr>& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
      if (!prim::ExprDeepEqual()(lhs[i], rhs[i])) {
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
    auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = non_inlined_bindings_.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    condition = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(condition, f_substitute)
                    .as_or_throw<PrimExpr>();
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

PrimFunc StmtSimplify(PrimFunc func, const arith::Analyzer& analyzer) {
  return arith::StmtSimplifier::Apply(std::move(func), analyzer);
}

namespace transform {

Pass StmtSimplify() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    arith::Analyzer analyzer;
    auto cfg = ctx->GetConfig<arith::StmtSimplifyConfig>("tirx.StmtSimplify");

    return arith::StmtSimplifier::Apply(f, analyzer, cfg);
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
