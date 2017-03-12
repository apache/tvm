/*!
 *  Copyright (c) 2017 by Contributors
 * \file split_pipeline.cc
 * \brief Split statement into pipeline stage modules.
 */
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/channel.h>
#include <unordered_map>
#include "./ir_util.h"

namespace tvm {
namespace ir {

class MarkChannelAccess : public IRMutator {
 public:
  MarkChannelAccess(
      const std::unordered_map<const Variable*, Channel>& cmap)
      : cmap_(cmap) {}

  Expr Mutate_(const Load *op, const Expr& e) final {
    auto it = rmap_.find(op->buffer_var.get());
    if (it != rmap_.end()) {
      ++it->second.read_count;
    }
    return IRMutator::Mutate_(op, e);
  }
  Stmt Mutate_(const Store *op, const Stmt& s) final {
    auto it = rmap_.find(op->buffer_var.get());
    if (it != rmap_.end()) {
      ++it->second.write_count;
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    if (cmap_.count(op->buffer_var.get())) {
      CHECK(!rmap_.count(op->buffer_var.get()));
      rmap_[op->buffer_var.get()] = Entry();
      Stmt body = Mutate(op->body);
      body = CreateChannelAccess(op, body);
      rmap_.erase(op->buffer_var.get());
      return body;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->type_key == ir::attr::storage_scope) {
      Var buf_var(op->node.node_);
      if (cmap_.count(buf_var.get())) return Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  // Create channel access wrap
  Stmt CreateChannelAccess(const Allocate* op, Stmt body) {
    const Entry& rw = rmap_.at(op->buffer_var.get());
    CHECK(rw.write_count == 0 || rw.read_count == 0)
        << "Cannot read/write to the same channel " << op->buffer_var
        <<  " body:" << body;
    if (rw.write_count == 0 && rw.read_count == 0) {
      return body;
    }
    const Channel& ch = cmap_.at(op->buffer_var.get());
    int32_t csize = op->constant_allocation_size();
    Expr alloc_size;
    if (csize > 0) {
      alloc_size = IntImm::make(Int(32), csize);
    } else {
      alloc_size = op->extents[0];
      for (size_t i = 1; i < op->extents.size(); ++i) {
        alloc_size *= op->extents[i];
      }
      alloc_size = ir::Simplify(alloc_size);
    }

    if (rw.write_count) {
      return AttrStmt::make(
          ch, ir::attr::channel_write_scope, alloc_size, body);
    } else {
      CHECK(rw.read_count);
      return AttrStmt::make(
          ch, ir::attr::channel_read_scope, alloc_size, body);
    }
  }
  struct Entry {
    int read_count{0};
    int write_count{0};
  };
  // The channels of each allocation.
  const std::unordered_map<const Variable*, Channel>& cmap_;
  // the result.
  std::unordered_map<const Variable*, Entry> rmap_;
};


// Mark the statment of each stage.
class StageSplitter : public IRMutator {
 public:
  Stmt Mutate(Stmt stmt) final {
    nest_.push_back(stmt);
    Stmt ret = IRMutator::Mutate(stmt);
    nest_.pop_back();
    return ret;
  }
  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) {
    if (!op->is_producer) return IRMutator::Mutate_(op, s);
    Stmt body = Mutate(op->body);
    stages_.emplace_back(BuildStage(body, op->func));
    return Evaluate::make(0);
  }

  Stmt Split(Stmt stmt) {
    stmt = Mutate(stmt);
    stmt = RemoveNoOp(stmt);
    CHECK(is_no_op(stmt));
    CHECK_NE(stages_.size(), 0);
    stmt = stages_.back();
    for (size_t i = stages_.size() - 1; i != 0; --i) {
      stmt = Block::make(stages_[i - 1], stmt);
    }
    stmt = MarkChannelAccess(cmap_).Mutate(stmt);
    return RemoveNoOp(stmt);
  }

 private:
  // Build the stage.
  Stmt BuildStage(Stmt body, NodeRef target) {
    int stage_index = static_cast<size_t>(stages_.size());
    std::string stage_suffix = "." + std::to_string(stage_index);
    // The Substitute
    Map<Var, Expr> subst;
    std::vector<Stmt> nest;
    Stmt no_op = Evaluate::make(0);

    for (const Stmt& s : nest_) {
      if (const For* op = s.as<For>()) {
        Var loop_var(op->loop_var);
        Var new_var = loop_var.copy_with_suffix(stage_suffix);
        subst.Set(loop_var, new_var);
        nest.emplace_back(For::make(
            new_var, op->min, op->extent,
            op->for_type, op->device_api, no_op));
      } else if (const LetStmt* op = s.as<LetStmt>()) {
        Var var(op->var);
        Var new_var = var.copy_with_suffix(stage_suffix);
        subst.Set(var, new_var);
        nest.emplace_back(LetStmt::make(new_var, op->value, no_op));
      } else if (const IfThenElse* op = s.as<IfThenElse>()) {
        CHECK(!op->else_case.defined());
        nest.emplace_back(IfThenElse::make(op->condition, no_op));
      } else if (const AttrStmt* op = s.as<AttrStmt>()) {
        nest.emplace_back(AttrStmt::make(
            op->node, op->type_key, op->value, no_op));
      } else if (s.as<ProducerConsumer>()) {
      } else if (s.as<Block>()) {
      } else if (const Allocate* op = s.as<Allocate>()) {
        nest.emplace_back(Allocate::make(
            op->buffer_var, op->type, op->extents,
            op->condition, no_op, op->new_expr, op->free_function));
        MarkChannel(op);
      } else {
        LOG(FATAL) << "not supported nest type " << s->type_key();
      }
    }
    body = Substitute(MergeNest(nest, body), subst);
    return AttrStmt::make(
        target, ir::attr::pipeline_stage_scope,
        make_const(Int(32), stage_index), body);
  }
  void MarkChannel(const Allocate* op) {
    if (!cmap_.count(op->buffer_var.get())) {
      Channel ch = ChannelNode::make(Var(op->buffer_var), op->type);
      cmap_[op->buffer_var.get()] = ch;
    }
  }
  // The stack
  std::vector<Stmt> nest_;
  // The stages
  std::vector<Stmt> stages_;
  // channel map
  std::unordered_map<const Variable*, Channel> cmap_;
};

Stmt SplitPipeline(Stmt stmt) {
  return StageSplitter().Split(stmt);
}

}  // namespace ir
}  // namespace tvm
