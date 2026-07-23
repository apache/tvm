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

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

namespace tvm {
namespace tirx {

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

// Is an Evaluate of tvm_call_packed("runtime.cuTensorMapEncodeTiled", ...)?
inline const CallNode* AsCuTensorMapEncode(const EvaluateNode* eval) {
  const CallNode* call = eval->value.as<CallNode>();
  if (!call || !call->op.same_as(builtin::tvm_call_packed())) return nullptr;
  if (call->args.empty()) return nullptr;
  if (const auto* s = call->args[0].as<StringImmNode>()) {
    if (s->value == "runtime.cuTensorMapEncodeTiled") return call;
  }
  return nullptr;
}

// Extract the tensormap var and the key (arguments after the tensormap var)
inline std::pair<ffi::Optional<Var>, ffi::Array<Expr>> ExtractEncodeKey(const CallNode* call) {
  TVM_FFI_ICHECK(call->op.same_as(builtin::tvm_call_packed()));
  // args[0] is function name, args[1] is tensormap handle, rest are parameters
  if (call->args.size() < 2) return {ffi::Optional<Var>(), ffi::Array<Expr>()};
  ffi::Optional<Var> tensormap;
  if (auto v = call->args[1].as<Var>()) {
    tensormap = v.value();
  } else {
    tensormap = ffi::Optional<Var>();
  }
  ffi::Array<Expr> key;
  key.reserve(call->args.size() - 2);
  for (size_t i = 2; i < call->args.size(); ++i) {
    key.push_back(call->args[i]);
  }
  return {tensormap, key};
}

}  // namespace

// First pass: Analyze encode calls and decide canonical tensormap per-parameter set
class CuTensorMapDedupAnalyzer : public StmtExprVisitor {
 public:
  CuTensorMapDedupAnalyzer() {
    canonical_list_.emplace_back(std::vector<std::pair<ffi::Array<Expr>, Var>>());
  }

  void VisitStmt_(const ForNode* op) final {
    StmtExprVisitor::VisitExpr(op->min);
    StmtExprVisitor::VisitExpr(op->extent);
    canonical_list_.emplace_back(std::vector<std::pair<ffi::Array<Expr>, Var>>());
    StmtExprVisitor::VisitStmt(op->body);
    canonical_list_.pop_back();
  }

  void VisitStmt_(const WhileNode* op) final {
    StmtExprVisitor::VisitExpr(op->condition);
    canonical_list_.emplace_back(std::vector<std::pair<ffi::Array<Expr>, Var>>());
    StmtExprVisitor::VisitStmt(op->body);
    canonical_list_.pop_back();
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    StmtExprVisitor::VisitExpr(op->condition);
    canonical_list_.emplace_back(std::vector<std::pair<ffi::Array<Expr>, Var>>());
    StmtExprVisitor::VisitStmt(op->then_case);
    canonical_list_.pop_back();
    if (op->else_case) {
      canonical_list_.emplace_back(std::vector<std::pair<ffi::Array<Expr>, Var>>());
      StmtExprVisitor::VisitStmt(op->else_case.value());
      canonical_list_.pop_back();
    }
  }

  void VisitStmt_(const EvaluateNode* op) final {
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
                var_remap_[v] = canonical;
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
    StmtExprVisitor::VisitStmt_(op);
  }

  const std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>& var_remap() const {
    return var_remap_;
  }

 private:
  std::vector<std::vector<std::pair<ffi::Array<Expr>, Var>>> canonical_list_;
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap_;
};

// Second pass: Rewrite vars to canonical, remove duplicate allocas and duplicate encode calls
class CuTensorMapDedupRewriter : public StmtExprMutator {
 public:
  CuTensorMapDedupRewriter(
      std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap)
      : var_remap_(std::move(var_remap)) {
    emitted_keys_.emplace_back(std::vector<ffi::Array<Expr>>());
  }

 private:
  using StmtExprMutator::VisitExpr_;
  using StmtExprMutator::VisitStmt_;

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    ffi::Array<Stmt> seq;
    seq.reserve(op->seq.size());
    bool changed = false;
    for (const Stmt& stmt : op->seq) {
      Stmt new_stmt = VisitStmt(stmt);
      // Dropped statements are represented as Evaluate(0).
      if (const auto* eval = new_stmt.as<EvaluateNode>()) {
        auto value = eval->value.as<PrimExpr>();
        if (value && is_zero(value.value())) {
          changed = true;
          continue;
        }
      }
      if (!new_stmt.same_as(stmt)) {
        changed = true;
      }
      seq.push_back(std::move(new_stmt));
    }
    if (!changed) {
      return ffi::GetRef<Stmt>(op);
    }
    return SeqStmt::Flatten(seq);
  }

  Expr VisitExpr_(const VarNode* op) final {
    Var v = ffi::GetRef<Var>(op);
    auto it = var_remap_.find(v);
    if (it != var_remap_.end()) {
      return it->second;
    }
    return v;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    PrimExpr min = VisitPrimExpr(op->min);
    PrimExpr extent = VisitPrimExpr(op->extent);
    emitted_keys_.emplace_back(std::vector<ffi::Array<Expr>>());
    Stmt body = VisitStmt(op->body);
    emitted_keys_.pop_back();
    if (min.same_as(op->min) && extent.same_as(op->extent) && body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->min = std::move(min);
      n->extent = std::move(extent);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const WhileNode* op) {
    PrimExpr condition = VisitPrimExpr(op->condition);
    emitted_keys_.emplace_back(std::vector<ffi::Array<Expr>>());
    Stmt body = VisitStmt(op->body);
    emitted_keys_.pop_back();
    if (condition.same_as(op->condition) && body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->condition = std::move(condition);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const IfThenElseNode* op) {
    PrimExpr condition = VisitPrimExpr(op->condition);
    emitted_keys_.emplace_back(std::vector<ffi::Array<Expr>>());
    Stmt then_case = VisitStmt(op->then_case);
    emitted_keys_.pop_back();
    ffi::Optional<Stmt> else_case = std::nullopt;
    if (op->else_case) {
      emitted_keys_.emplace_back(std::vector<ffi::Array<Expr>>());
      else_case = VisitStmt(op->else_case.value());
      emitted_keys_.pop_back();
    }
    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->condition = std::move(condition);
      n->then_case = std::move(then_case);
      n->else_case = std::move(else_case);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const BindNode* op) final {
    Expr value = VisitExpr(op->value);
    if (IsTensorMapAlloca(op)) {
      // If this bind allocates a tensormap that is remapped to a canonical var, drop it.
      auto it = var_remap_.find(op->var);
      if (it != var_remap_.end()) {
        return Evaluate(0);
      }
    }
    if (value.same_as(op->value)) {
      return ffi::GetRef<Stmt>(op);
    }
    return Bind(op->var, value, op->span);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    // Default mutation
    Evaluate eval = StmtExprMutator::VisitStmt_(op).as_or_throw<Evaluate>();
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

  // Map of duplicate var -> canonical var
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap_;
  // Track which parameter keys have already emitted an encode call
  std::vector<std::vector<ffi::Array<Expr>>> emitted_keys_;
};

namespace transform {

Pass LowerTIRxDedupCuTensorMaps() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    // Analyze usage to find duplicates
    CuTensorMapDedupAnalyzer analyzer;
    analyzer(f->body);
    if (analyzer.var_remap().empty()) {
      return f;
    }
    auto* n = f.CopyOnWrite();
    n->body = CuTensorMapDedupRewriter(analyzer.var_remap())(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxDedupCuTensorMaps", {});
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
