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
 *  Loop unrolling as in Halide pipeline.
 * \file unroll_loop.cc
 */
// Unrolls the loop as in Halide pipeline.
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

struct UnrollLoopConfigNode : public tvm::AttrsNode<UnrollLoopConfigNode> {
  int auto_max_step;
  int auto_max_depth;
  int auto_max_extent;
  int explicit_unroll;
  int unroll_local_access;

  TVM_DECLARE_ATTRS(UnrollLoopConfigNode, "tir.transform.UnrollLoopConfig") {
    TVM_ATTR_FIELD(auto_max_step)
        .describe("Threshold of number of steps in the loop to be automatically unrolled")
        .set_default(0);
    TVM_ATTR_FIELD(auto_max_depth)
        .describe("The maximum nested level of loops that can be automatically unrolled.")
        .set_default(8);
    TVM_ATTR_FIELD(auto_max_extent)
        .describe("The maximum extent of loop that will be unrolled.")
        .set_default(0);
    TVM_ATTR_FIELD(explicit_unroll)
        .describe("Whether to explicitly unroll the loop instead of setting a pragma")
        .set_default(true);
    TVM_ATTR_FIELD(unroll_local_access)
        .describe("Whether to always unroll local access")
        .set_default(false);
  }
};

class UnrollLoopConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(UnrollLoopConfig, Attrs, UnrollLoopConfigNode);
};

TVM_REGISTER_NODE_TYPE(UnrollLoopConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.UnrollLoop", UnrollLoopConfig);

class VarLocalAccessMarker : public ExprVisitor {
 public:
  explicit VarLocalAccessMarker(std::unordered_set<Var>* var_touched_local)
      : var_touched_local_(var_touched_local) {}

  void VisitExpr_(const VarNode* op) final { var_touched_local_->insert(GetRef<Var>(op)); }

 private:
  std::unordered_set<Var>* var_touched_local_;
};

// The Visitor is used to check whether var is used as write index in a local memory
// If a loop var is used as indices to a local memory, it must be unrolled so
// the local memory access can be turned into register access.
class LoopUnroller : public StmtExprMutator {
 public:
  explicit LoopUnroller(int auto_max_step, int auto_max_depth, int auto_max_extent,
                        bool explicit_unroll, bool unroll_local_access)
      : auto_max_step_(auto_max_step),
        auto_max_depth_(auto_max_depth),
        auto_max_extent_(auto_max_extent),
        explicit_unroll_(explicit_unroll),
        unroll_local_access_(unroll_local_access) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == "pragma_auto_unroll_max_step") {
      int value = static_cast<int>(Downcast<Integer>(op->value)->value);
      std::swap(value, auto_max_step_);
      Stmt ret = this->VisitStmt(op->body);
      std::swap(value, auto_max_step_);
      return ret;
    } else if (op->attr_key == "pragma_unroll_explicit") {
      bool explicit_unroll = Downcast<Integer>(op->value)->value;
      std::swap(explicit_unroll, explicit_unroll_);
      Stmt ret = this->VisitStmt(op->body);
      std::swap(explicit_unroll, explicit_unroll_);
      return ret;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const ForNode* op) {
    // Post order so we can collect more information
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<ForNode>();
    int value = GetExtent(op);
    // condition for auto unroll
    bool auto_unroll = (op->kind == ForKind::kSerial && value >= 0 && normal_loop_depth_ == 0 &&
                        unroll_depth_ <= auto_max_depth_);

    auto_unroll =
        auto_unroll && (value * step_count_ <= auto_max_step_ || value <= auto_max_extent_);

    if (op->kind == ForKind::kUnrolled) {
      ICHECK_GE(value, 0) << "Cannot unroll non-constant loop";
      auto_unroll = true;
    }

    // If a loop var is used as indices to a local memory, it must be unrolled so
    // the local memory access can be turned into register access.
    if (this->var_touched_local_.count(op->loop_var) && value > 0 && unroll_local_access_) {
      auto_unroll = true;
    }

    if (auto_unroll) {
      step_count_ *= value;
      unroll_depth_ += 1;
    } else {
      normal_loop_depth_ += 1;
    }

    if ((auto_unroll && explicit_unroll_) ||
        // unroll loops with extent = 1, no matter how many steps in body
        (0 <= value && value <= auto_max_extent_ && auto_max_extent_ == 1)) {
      return Unroll(op);
    } else {
      if (auto_unroll) {
        if (op->kind != ForKind::kUnrolled) {
          return For(op->loop_var, op->min, op->extent, ForKind::kUnrolled, op->body,
                     op->thread_binding, op->annotations);
        }
      }
      return stmt;
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (unroll_local_access_) {
      auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer->data));
      if (storage_scope.rank == runtime::StorageRank::kLocal ||
          storage_scope.rank == runtime::StorageRank::kWarp) {
        VarLocalAccessMarker marker(&var_touched_local_);
        for (PrimExpr e : op->indices) {
          marker(e);
        }
      }
    }
    return GetRef<PrimExpr>(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    ++step_count_;
    if (unroll_local_access_) {
      auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer->data));
      if (storage_scope.rank == runtime::StorageRank::kLocal ||
          storage_scope.rank == runtime::StorageRank::kWarp) {
        VarLocalAccessMarker marker(&var_touched_local_);
        for (PrimExpr e : op->indices) {
          marker(e);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    ++step_count_;
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    auto fmutate = [this](const Stmt& s) {
      int step_count = step_count_;
      int unroll_depth = unroll_depth_;
      int normal_loop_depth = normal_loop_depth_;
      step_count_ = 0;
      unroll_depth_ = 0;
      normal_loop_depth_ = 0;
      Stmt ret = this->VisitStmt(s);
      step_count_ += step_count;
      normal_loop_depth_ = std::max(normal_loop_depth, normal_loop_depth_);
      unroll_depth_ = std::max(unroll_depth_, unroll_depth);
      return ret;
    };
    return StmtExprMutator::VisitSeqStmt_(op, false, fmutate);
  }

  Stmt Unroll(const ForNode* op) {
    int value = GetExtent(op);
    // For loop must have a constant integer extent
    ICHECK_NE(value, -1) << "loop doesn't have a constant integer extent";
    if (value == 0) return Evaluate(0);
    Stmt body = op->body;
    Map<Var, PrimExpr> vmap;
    Array<Stmt> unrolled;
    for (int i = 0; i < value; ++i) {
      vmap.Set(op->loop_var, op->min + make_const(op->loop_var.dtype(), i));
      Stmt step = Substitute(body, vmap);
      unrolled.push_back(step);
    }
    return SeqStmt::Flatten(unrolled);
  }

 private:
  // returns the extent of the loop if it's a constant integer, otherwise return -1
  int GetExtent(const ForNode* op) {
    // constant folding.
    PrimExpr extent = analyzer_.Simplify(op->extent);
    const IntImmNode* v1 = extent.as<IntImmNode>();
    int value = -1;
    // integers that do not fit in int32_t are treated as symbolic,
    // as it's impossible to unroll such large loops
    if (v1 != nullptr && v1->value <= std::numeric_limits<int>::max()) {
      value = static_cast<int>(v1->value);
    }
    return value;
  }

  // maximum number of step to perform auto unroll.
  int auto_max_step_;
  int auto_max_depth_;
  // max extent of loop to auto unroll
  // this does not count the total steps, only count the number of loops
  int auto_max_extent_;
  bool explicit_unroll_;
  // Wether to unroll loops to local access.
  bool unroll_local_access_{false};
  // Number of normal loops in scope
  int normal_loop_depth_{0};
  // number of unrolled cases in current scope.
  int unroll_depth_{0};
  // Number of total steps unrolled
  int step_count_{0};
  // set of indices touched during visit local memory
  std::unordered_set<Var> var_touched_local_;
  // analyzer
  arith::Analyzer analyzer_;
};

Stmt UnrollLoop(Stmt stmt, UnrollLoopConfig cfg) {
  Stmt ret = LoopUnroller(cfg->auto_max_step, cfg->auto_max_depth, cfg->auto_max_extent,
                          cfg->explicit_unroll, cfg->unroll_local_access)(stmt);
  if (!ret.same_as(stmt)) {
    return ConvertSSA(ret);
  } else {
    return ret;
  }
}

namespace transform {

Pass UnrollLoop() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto cfg = ctx->GetConfig<UnrollLoopConfig>("tir.UnrollLoop");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<UnrollLoopConfig>();
    }
    n->body = UnrollLoop(std::move(f->body), cfg.value());
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.UnrollLoop", {});
}

TVM_REGISTER_GLOBAL("tir.transform.UnrollLoop").set_body_typed(UnrollLoop);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
