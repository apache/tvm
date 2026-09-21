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
 * \file lower_tirx_dedup_tensormap.cc
 * \brief Deduplicate identical cuTensorMap objects created by TIRx schedules.
 */

#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

namespace {

// Helper to check if a call is to tvm.tir builtin op
inline bool IsBuiltin(const CallNode* call, const Op& op) { return call && call->op.same_as(op); }

// Is a stack allocation for a tensormap handle?
inline bool IsTensorMapAlloca(const BindNode* bind) {
  if (const auto* call = bind->value.as<CallNode>()) {
    if (IsBuiltin(call, builtin::tvm_stack_alloca())) {
      if (call->args.size() == 2) {
        if (const auto* type_str = call->args[0].as<StringImmNode>()) {
          return type_str->value == "tensormap";
        }
      }
    }
  }
  return false;
}

// Recognize typed encoding and legacy manually authored packed encoding.
inline const CallNode* AsCuTensorMapEncode(const EvaluateNode* eval) {
  const CallNode* call = eval->value.as<CallNode>();
  if (!call) return nullptr;
  if (call->op.same_as(builtin::tensormap_encode_tiled())) return call;
  if (!call->op.same_as(builtin::tvm_call_packed())) return nullptr;
  if (call->args.empty()) return nullptr;
  if (const auto* s = call->args[0].as<StringImmNode>()) {
    if (s->value == "runtime.cuTensorMapEncodeTiled") return call;
  }
  return nullptr;
}

// Exclude only the output pointer; retain op, attributes and all input operands
// so descriptor dtype, forced dtype and encoding modes participate in equality.
inline std::pair<ffi::Optional<Var>, Call> ExtractEncodeKey(const CallNode* call) {
  size_t output_index = call->op.same_as(builtin::tensormap_encode_tiled()) ? 0 : 1;
  TVM_FFI_ICHECK_GT(call->args.size(), output_index);
  ffi::Optional<Var> tensormap = call->args[output_index].as<Var>();
  ffi::Array<Expr> args;
  for (size_t i = 0; i < call->args.size(); ++i) {
    if (i != output_index) args.push_back(call->args[i]);
  }
  return {tensormap, Call(call->ty, call->op, args, call->attrs)};
}

}  // namespace

// First pass: Analyze encode calls and decide canonical tensormap per-parameter set
class CuTensorMapDedupAnalyzer : public StmtExprVisitor {
 public:
  CuTensorMapDedupAnalyzer() { canonical_list_.emplace_back(std::vector<std::pair<Call, Var>>()); }

  ffi::Optional<VisitInterrupt> Visit_(const ForNode* op) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->min));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->extent));
    canonical_list_.emplace_back(std::vector<std::pair<Call, Var>>());
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->body));
    canonical_list_.pop_back();
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const WhileNode* op) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->condition));
    canonical_list_.emplace_back(std::vector<std::pair<Call, Var>>());
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->body));
    canonical_list_.pop_back();
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const IfThenElseNode* op) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->condition));
    canonical_list_.emplace_back(std::vector<std::pair<Call, Var>>());
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->then_case));
    canonical_list_.pop_back();
    if (op->else_case) {
      canonical_list_.emplace_back(std::vector<std::pair<Call, Var>>());
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->else_case.value()));
      canonical_list_.pop_back();
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const EvaluateNode* op) final {
    if (const CallNode* call = AsCuTensorMapEncode(op)) {
      auto [maybe_var, key] = ExtractEncodeKey(call);
      if (maybe_var.has_value()) {
        const Var& v = maybe_var.value();
        // Find an existing key that is structurally equal
        bool found = false;
        for (const auto& sub_canonical_list : canonical_list_) {
          for (const auto& kv : sub_canonical_list) {
            if (ffi::StructuralEqual()(kv.first, key)) {
              const Var& canonical = kv.second;
              if (!canonical.same_as(v)) {
                tensormap_var_remap_[v] = canonical;
              }
              found = true;
              break;
            }
          }
          if (found) break;
        }
        if (!found) canonical_list_.back().emplace_back(std::move(key), v);
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  const std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& var_remap() const {
    return tensormap_var_remap_;
  }

 private:
  std::vector<std::vector<std::pair<Call, Var>>> canonical_list_;
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> tensormap_var_remap_;
};

// Second pass: Rewrite vars to canonical, remove duplicate allocas and duplicate encode calls
class CuTensorMapDedupRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  CuTensorMapDedupRewriter(
      std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap) {
    for (const auto& [source, target] : var_remap) VarRemapSet(source, target);
    emitted_keys_.emplace_back(std::vector<Call>());
  }

 private:
  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    auto min_result = Mutate(op->min, inplace_mode);
    bool min_unchanged = min_result.UnchangedOrSameAs(op->min);
    PrimExpr min = std::move(min_result).ValueOrUnchanged(op->min);
    auto extent_result = Mutate(op->extent, inplace_mode);
    bool extent_unchanged = extent_result.UnchangedOrSameAs(op->extent);
    PrimExpr extent = std::move(extent_result).ValueOrUnchanged(op->extent);
    emitted_keys_.emplace_back();
    auto body_result = Mutate(op->body, inplace_mode);
    emitted_keys_.pop_back();
    bool body_unchanged = body_result.UnchangedOrSameAs(op->body);
    Stmt body = std::move(body_result).ValueOrUnchanged(op->body);
    if (min_unchanged && extent_unchanged && body_unchanged) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* n = const_cast<ForNode*>(op);
        n->min = std::move(min);
        n->extent = std::move(extent);
        n->body = std::move(body);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<ForNode>(*op);
      n->min = std::move(min);
      n->extent = std::move(extent);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  UnchangedOr<Stmt> Mutate_(const WhileNode* op, InplaceMode inplace_mode) {
    auto condition_result = Mutate(op->condition, inplace_mode);
    bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
    PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
    emitted_keys_.emplace_back();
    auto body_result = Mutate(op->body, inplace_mode);
    emitted_keys_.pop_back();
    bool body_unchanged = body_result.UnchangedOrSameAs(op->body);
    Stmt body = std::move(body_result).ValueOrUnchanged(op->body);
    if (condition_unchanged && body_unchanged) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* n = const_cast<WhileNode*>(op);
        n->condition = std::move(condition);
        n->body = std::move(body);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<WhileNode>(*op);
      n->condition = std::move(condition);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  UnchangedOr<Stmt> Mutate_(const IfThenElseNode* op, InplaceMode inplace_mode) {
    auto condition_result = Mutate(op->condition, inplace_mode);
    bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
    PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
    emitted_keys_.emplace_back();
    auto then_case_result = Mutate(op->then_case, inplace_mode);
    emitted_keys_.pop_back();
    bool then_case_unchanged = then_case_result.UnchangedOrSameAs(op->then_case);
    Stmt then_case = std::move(then_case_result).ValueOrUnchanged(op->then_case);
    ffi::Optional<Stmt> else_case = std::nullopt;
    if (op->else_case) {
      emitted_keys_.emplace_back();
      else_case =
          Mutate(op->else_case.value(), inplace_mode).ValueOrUnchanged(op->else_case.value());
      emitted_keys_.pop_back();
    }
    if (condition_unchanged && then_case_unchanged && else_case.same_as(op->else_case)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* n = const_cast<IfThenElseNode*>(op);
        n->condition = std::move(condition);
        n->then_case = std::move(then_case);
        n->else_case = std::move(else_case);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<IfThenElseNode>(*op);
      n->condition = std::move(condition);
      n->then_case = std::move(then_case);
      n->else_case = std::move(else_case);
      return Stmt(n);
    }
  }

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    auto value_result = Mutate(op->value, inplace_mode);
    bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
    Expr value = std::move(value_result).ValueOrUnchanged(op->value);
    if (IsTensorMapAlloca(op)) {
      // If this bind allocates a tensormap that is remapped to a canonical var, drop it.
      if (VarRemapGet(op->var) != nullptr) {
        return Evaluate(0);
      }
    }
    if (value_unchanged) {
      return ffi::Unchanged();
    }
    return Bind(op->var, value, op->span);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode* op, InplaceMode inplace_mode) final {
    // Default mutation
    Evaluate eval = StmtExprMutator::Mutate_(op, inplace_mode)
                        .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                        .as_or_throw<Evaluate>();
    if (const CallNode* call = AsCuTensorMapEncode(eval.get())) {
      // Build key after var remapping
      auto [maybe_var, key] = ExtractEncodeKey(call);
      // Keep only the first occurrence for this key in the frame
      for (const auto& sub_emitted_keys : emitted_keys_) {
        for (const auto& k : sub_emitted_keys) {
          if (ffi::StructuralEqual()(k, key)) {
            return Evaluate(0);
          }
        }
      }
      emitted_keys_.back().emplace_back(std::move(key));
      return eval;
    }
    return eval;
  }

  // Track which parameter keys have already emitted an encode call
  std::vector<std::vector<Call>> emitted_keys_;
};

namespace transform {

Pass LowerTIRxDedupCuTensorMaps() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    // Analyze usage to find duplicates
    auto analyzer = ffi::make_object<CuTensorMapDedupAnalyzer>();
    analyzer->Visit(f->body);
    if (analyzer->var_remap().empty()) {
      return f;
    }
    auto* n = f.CopyOnWrite();
    n->body = ffi::make_object<CuTensorMapDedupRewriter>(analyzer->var_remap())
                  ->Mutate(n->body, InplaceMode::kAllow)
                  .ValueOrUnchanged(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxDedupCuTensorMaps", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
