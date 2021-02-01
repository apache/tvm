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
 * \brief Inject double buffering optimization for data fetch.
 * \file inject_double_buffer.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

struct InjectDoubleBufferConfigNode : public tvm::AttrsNode<InjectDoubleBufferConfigNode> {
  int split_loop;

  TVM_DECLARE_ATTRS(InjectDoubleBufferConfigNode, "tir.transform.InjectDoubleBufferConfig") {
    TVM_ATTR_FIELD(split_loop).describe("Split loop factors").set_default(1);
  }
};

class InjectDoubleBufferConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(InjectDoubleBufferConfig, Attrs,
                                            InjectDoubleBufferConfigNode);
};

TVM_REGISTER_NODE_TYPE(InjectDoubleBufferConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.InjectDoubleBuffer", InjectDoubleBufferConfig);

// Detect double buffer variables.
class DoubleBufferDetector : public StmtExprVisitor {
 public:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::double_buffer_scope) {
      touched_.insert(op->node.as<VarNode>());
      StmtExprVisitor::VisitStmt_(op);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitExpr_(const VarNode* op) final {
    if (touched_.count(op)) {
      touched_.erase(op);
    }
  }
  // The set of touched variable.
  std::unordered_set<const VarNode*> touched_;
};

class StripDoubleBufferWrite : public StmtMutator {
 public:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::double_buffer_write) {
      return VisitStmt(op->body);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }
};

class DoubleBufferInjector : public StmtExprMutator {
 public:
  explicit DoubleBufferInjector(int split_loop) : split_loop_(split_loop) {}

  Stmt Inject(Stmt stmt) {
    DoubleBufferDetector detector;
    detector(stmt);
    if (detector.touched_.empty()) return stmt;
    for (const VarNode* v : detector.touched_) {
      dbuffer_info_[v] = StorageEntry();
    }
    return ConvertSSA(operator()(std::move(stmt)));
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::storage_scope) {
      const VarNode* buf = op->node.as<VarNode>();
      auto it = dbuffer_info_.find(buf);
      if (it != dbuffer_info_.end()) {
        it->second.scope = op->value.as<StringImmNode>()->value;
        return this->VisitStmt(op->body);
      } else {
        return StmtExprMutator::VisitStmt_(op);
      }
    } else if (op->attr_key == attr::double_buffer_scope) {
      return MakeProducer(op);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    auto it = dbuffer_info_.find(op->buffer_var.get());
    if (it != dbuffer_info_.end()) {
      it->second.stride = foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                                make_const(DataType::Int(32), 1), op->extents) *
                          op->dtype.lanes();
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<AllocateNode>();
      Array<PrimExpr> new_extents{make_const(op->extents[0].dtype(), 2)};
      for (PrimExpr e : op->extents) {
        new_extents.push_back(e);
      }
      ICHECK(it->second.loop != nullptr);
      auto& alloc_nest = loop_allocs_[it->second.loop];
      alloc_nest.emplace_back(
          AttrStmt(op->buffer_var, attr::storage_scope, StringImm(it->second.scope), Evaluate(0)));
      alloc_nest.emplace_back(
          Allocate(op->buffer_var, op->dtype, new_extents, op->condition, Evaluate(0)));
      return op->body;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    loop_nest_.push_back(op);
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    auto it = loop_pre_.find(op);
    if (it != loop_pre_.end()) {
      const ForNode* old_loop = stmt.as<ForNode>();
      if (split_loop_ != 0) {
        // Explicitly unroll the loop
        ICHECK(split_loop_ % 2 == 0 || split_loop_ == 1)
            << "It is better to split with multiple of 2";
        ICHECK(is_zero(old_loop->min));
        PrimExpr zero = old_loop->min;
        PrimExpr new_ext = old_loop->extent - make_const(old_loop->loop_var.dtype(), 1);
        PrimExpr factor = make_const(new_ext.dtype(), split_loop_);
        PrimExpr outer_ext = new_ext / factor;
        PrimExpr tail_base = outer_ext * factor;
        Var outer_var(old_loop->loop_var->name_hint + ".outer", old_loop->loop_var.dtype());
        std::unordered_map<const VarNode*, PrimExpr> vmap;
        std::vector<Stmt> loop_seq;
        for (int32_t i = 0; i < split_loop_; ++i) {
          vmap[old_loop->loop_var.get()] = outer_var * factor + make_const(factor.dtype(), i);
          loop_seq.emplace_back(Substitute(old_loop->body, vmap));
        }
        Stmt loop = For(outer_var, zero, outer_ext, old_loop->kind, SeqStmt::Flatten(loop_seq));
        // tail
        std::vector<Stmt> tail_seq;
        Stmt tail_body = StripDoubleBufferWrite()(old_loop->body);
        for (int32_t i = 0; i < split_loop_; ++i) {
          PrimExpr idx = tail_base + make_const(tail_base.dtype(), i);
          vmap[old_loop->loop_var.get()] = idx;
          tail_seq.emplace_back(IfThenElse(idx < old_loop->extent, Substitute(tail_body, vmap)));
        }
        stmt = SeqStmt::Flatten(loop, tail_seq);
      }
      stmt = SeqStmt::Flatten(it->second, stmt);
    }
    it = loop_allocs_.find(op);
    if (it != loop_allocs_.end()) {
      stmt = MergeNest(it->second, stmt);
    }
    loop_nest_.pop_back();
    return stmt;
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    auto it = dbuffer_info_.find(op->buffer_var.get());
    if (it != dbuffer_info_.end()) {
      const StorageEntry& e = it->second;
      ICHECK(in_double_buffer_scope_);
      ICHECK(e.stride.defined());
      return Store(op->buffer_var, op->value, e.switch_write_var * e.stride + op->index,
                   op->predicate);
    } else {
      return stmt;
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    auto it = dbuffer_info_.find(op->buffer_var.get());
    if (it != dbuffer_info_.end()) {
      const StorageEntry& e = it->second;
      ICHECK(e.stride.defined());
      ICHECK(e.switch_read_var.defined());
      return Load(op->dtype, op->buffer_var, e.switch_read_var * e.stride + op->index,
                  op->predicate);
    } else {
      return expr;
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    ICHECK(!dbuffer_info_.count(op));
    return GetRef<PrimExpr>(op);
  }

 private:
  Stmt MakeProducer(const AttrStmtNode* op) {
    const Var buffer = Downcast<Var>(op->node);
    ICHECK_NE(loop_nest_.size(), 0U) << "Double buffer scope must be inside a loop";
    auto it = dbuffer_info_.find(buffer.get());
    if (it == dbuffer_info_.end()) {
      LOG(WARNING) << "Skip double buffer scope " << op->node;
      return this->VisitStmt(op->body);
    }
    StorageEntry& e = it->second;
    e.loop = loop_nest_.back();
    PrimExpr zero = make_const(e.loop->loop_var.dtype(), 0);
    PrimExpr one = make_const(e.loop->loop_var.dtype(), 1);
    PrimExpr two = make_const(e.loop->loop_var.dtype(), 2);
    PrimExpr loop_shift = e.loop->loop_var + one;
    e.switch_write_var = Var(e.loop->loop_var->name_hint + ".db", e.loop->loop_var.dtype());
    e.switch_read_var = indexmod(e.loop->loop_var, two);
    in_double_buffer_scope_ = true;
    Stmt body = this->VisitStmt(op->body);
    in_double_buffer_scope_ = false;
    std::unordered_map<const VarNode*, PrimExpr> vmap;
    vmap[e.switch_write_var.get()] = zero;
    vmap[e.loop->loop_var.get()] = zero;
    loop_pre_[e.loop].emplace_back(Substitute(body, vmap));
    vmap[e.loop->loop_var.get()] = loop_shift;
    vmap[e.switch_write_var.get()] = indexmod(loop_shift, two);
    body = Substitute(body, vmap);
    body = AttrStmt(buffer, attr::double_buffer_write, 1, body);
    body = IfThenElse(loop_shift < e.loop->extent, body);
    return body;
  }
  // Storage entry for those who need double buffering.
  struct StorageEntry {
    // The size of the buffer
    PrimExpr stride;
    // The loop we need
    const ForNode* loop{nullptr};
    // The switch variable.
    Var switch_write_var;
    // The switch variable for reading.
    PrimExpr switch_read_var;
    // The storage scope.
    std::string scope;
  };
  // Whether split loop
  int32_t split_loop_;
  // Whether we are inside double buffer scope.
  bool in_double_buffer_scope_{false};
  // The current loop next
  std::vector<const ForNode*> loop_nest_;
  // The allocs to be appended before the loop
  std::unordered_map<const ForNode*, std::vector<Stmt> > loop_allocs_;
  // The stmt to be appended before the loop
  std::unordered_map<const ForNode*, std::vector<Stmt> > loop_pre_;
  // The allocation size of the buffer
  std::unordered_map<const VarNode*, StorageEntry> dbuffer_info_;
};

namespace transform {

Pass InjectDoubleBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto cfg = ctx->GetConfig<InjectDoubleBufferConfig>("tir.InjectDoubleBuffer");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<InjectDoubleBufferConfig>();
    }
    n->body = DoubleBufferInjector(cfg.value()->split_loop).Inject(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectDoubleBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectDoubleBuffer").set_body_typed(InjectDoubleBuffer);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
