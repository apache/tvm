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

#include "stmt_simplify.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/transform.h>

#include "../ir/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

void StmtSimplifyConfigNode::RegisterReflection() {
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

sym::RewriteSimplifier::Extension StmtSimplifyConfigNode::GetEnabledExtensions() const {
  sym::RewriteSimplifier::Extension flags = sym::RewriteSimplifier::kNone;
  if (transitively_prove_inequalities) {
    flags = sym::RewriteSimplifier::Extension(
        flags | sym::RewriteSimplifier::kTransitivelyProveInequalities);
  }
  if (convert_boolean_to_and_of_ors) {
    flags = sym::RewriteSimplifier::Extension(flags |
                                              sym::RewriteSimplifier::kConvertBooleanToAndOfOrs);
  }
  if (apply_constraints_to_boolean_branches) {
    flags = sym::RewriteSimplifier::Extension(
        flags | sym::RewriteSimplifier::kApplyConstraintsToBooleanBranches);
  }
  return flags;
}

static StmtSimplifyConfig MakeDefaultStmtSimplifyConfig() {
  return tvm::transform::PassConfigWithDefaults<StmtSimplifyConfig>();
}

TVM_FFI_STATIC_INIT_BLOCK() { StmtSimplifyConfigNode::RegisterReflection(); }

TVM_REGISTER_PASS_CONFIG_OPTION("tirx.StmtSimplify", StmtSimplifyConfig);

PrimFunc StmtSimplifier::Apply(PrimFunc func, const sym::Analyzer& analyzer,
                               ffi::Optional<StmtSimplifyConfig> config_opt) {
  auto config = config_opt.value_or(MakeDefaultStmtSimplifyConfig());

  auto simplifier = ffi::make_object<StmtSimplifier>(analyzer, config);
  return simplifier->Run(std::move(func));
}

PrimFunc StmtSimplifier::Run(PrimFunc func) {
  analyzer_->rewrite_simplify.SetEnabledExtensions(config_->GetEnabledExtensions());
  MarkBufferParamShapes(func);
  auto* n = func.CopyOnWrite();
  n->body = Mutate(n->body, InplaceMode::kAllow).ValueOrUnchanged(n->body);
  return func;
}

UnchangedOr<ffi::Any> StmtSimplifier::Mutate(ffi::AnyView input, InplaceMode inplace_mode) {
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

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const ForNode* op, InplaceMode inplace_mode) {
  analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
  With<sym::ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
  With<sym::ConstraintContext> ctx2(analyzer_,
                                    static_cast<PrimExpr>(op->loop_var) < op->min + op->extent);
  return Parent::Mutate_(op, inplace_mode);
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const BindNode* op, InplaceMode inplace_mode) {
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

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const IfThenElseNode* op, InplaceMode inplace_mode) {
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

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) {
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

bool StmtSimplifier::ArrayDeepEqual(const ffi::Array<PrimExpr>& lhs,
                                    const ffi::Array<PrimExpr>& rhs) {
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

ffi::Optional<bool> StmtSimplifier::ProveCondition(PrimExpr condition) const {
  auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = non_inlined_bindings_.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  condition = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(condition, f_substitute)
                  .as_or_throw<PrimExpr>();
  condition = analyzer_->Simplify(condition);
  if (const auto* as_int = condition.as<IntImmNode>()) {
    return as_int->value != 0;
  } else {
    return std::nullopt;
  }
}

PrimFunc StmtSimplify(PrimFunc func, const sym::Analyzer& analyzer) {
  return StmtSimplifier::Apply(std::move(func), analyzer);
}

namespace transform {

Pass StmtSimplify() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    sym::Analyzer analyzer;
    auto cfg = ctx->GetConfig<StmtSimplifyConfig>("tirx.StmtSimplify");

    return StmtSimplifier::Apply(f, analyzer, cfg);
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
