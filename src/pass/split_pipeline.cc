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
#include <unordered_set>
#include "ir_util.h"

namespace tvm {
namespace ir {

class MarkChannelAccess : public IRMutator {
 public:
  MarkChannelAccess(
      const std::unordered_map<const Variable*, Channel>& cmap,
      const std::unordered_map<const Variable*, Channel>& fifo_map)
      : cmap_(cmap), fifo_map_(fifo_map) {}
  using IRMutator::Mutate;
  Stmt Mutate(Stmt stmt) final {
    Stmt ret = IRMutator::Mutate(stmt);
    if (read_fifos_.size() != 0) {
      for (const Variable* v : read_fifos_) {
        Channel ch = fifo_map_.at(v);
        ret = ReadChannel(ch, 1, ret);
      }
      read_fifos_.clear();
    }
    if (write_fifos_.size() != 0) {
      for (const Variable* v : write_fifos_) {
        Channel ch = fifo_map_.at(v);
        ret = WriteChannel(ch, 1, ret);
      }
      write_fifos_.clear();
    }
    return ret;
  }

  Expr Mutate_(const Load *op, const Expr& e) final {
    auto it = rmap_.find(op->buffer_var.get());
    if (it != rmap_.end()) {
      ++it->second.read_count;
    }
    if (fifo_map_.count(op->buffer_var.get())) {
      read_fifos_.insert(op->buffer_var.get());
      CHECK(!write_fifos_.count(op->buffer_var.get()));
    }
    return IRMutator::Mutate_(op, e);
  }
  Stmt Mutate_(const Store *op, const Stmt& s) final {
    auto it = rmap_.find(op->buffer_var.get());
    if (it != rmap_.end()) {
      ++it->second.write_count;
    }
    if (fifo_map_.count(op->buffer_var.get())) {
      write_fifos_.insert(op->buffer_var.get());
      CHECK(!read_fifos_.count(op->buffer_var.get()));
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
    if (op->attr_key == ir::attr::storage_scope) {
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
        alloc_size = alloc_size * op->extents[i];
      }
    }

    if (rw.write_count) {
      return WriteChannel(ch, alloc_size, body);
    } else {
      CHECK(rw.read_count);
      return ReadChannel(ch, alloc_size, body);
    }
  }
  Stmt ReadChannel(Channel ch, Expr size, Stmt body) {
    return AttrStmt::make(
        ch, ir::attr::channel_read_scope, size,
        AttrStmt::make(ch, ir::attr::channel_read_advance, size,
                       body));
  }
  Stmt WriteChannel(Channel ch, Expr size, Stmt body) {
    return AttrStmt::make(
        ch, ir::attr::channel_write_scope, size,
        AttrStmt::make(ch, ir::attr::channel_write_advance, size,
                       body));
  }
  struct Entry {
    int read_count{0};
    int write_count{0};
  };
  // The channels of each allocation.
  const std::unordered_map<const Variable*, Channel>& cmap_;
  // FIFO map.
  const std::unordered_map<const Variable*, Channel>& fifo_map_;
  // the result.
  std::unordered_map<const Variable*, Entry> rmap_;
  // Accessed FIFOs
  std::unordered_set<const Variable*> read_fifos_, write_fifos_;
};

// Mark the statment of each stage.
class StageSplitter : public IRMutator {
 public:
  using IRMutator::Mutate;
  explicit StageSplitter(bool split_load)
      : split_load_(split_load) {}

  Stmt Mutate(Stmt stmt) final {
    nest_.push_back(stmt);
    Stmt ret = IRMutator::Mutate(stmt);
    nest_.pop_back();
    return ret;
  }
  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) final {
    if (!op->is_producer) {
      return Mutate(op->body);
    }
    Stmt body = Mutate(op->body);
    stages_.emplace_back(BuildStage(body, op->func));
    return Evaluate::make(0);
  }
  Expr Mutate_(const Load* op, const Expr& e) final {
    if (!split_load_) return IRMutator::Mutate_(op, e);
    std::ostringstream cname;
    cname << "fifo." << temp_fifo_count_++;
    // Create FIFO channel for load.
    Channel ch = ChannelNode::make(Var(cname.str(), Handle()), op->type);
    Expr index = Mutate(op->index);
    Stmt provide = Store::make(
        ch->handle_var,
        Load::make(op->type, op->buffer_var, index, op->predicate),
        0, op->predicate);
    Stmt temp = nest_.back(); nest_.pop_back();
    stages_.emplace_back(BuildStage(provide, ch));
    nest_.push_back(temp);
    fifo_map_[ch->handle_var.get()] = ch;
    return Load::make(op->type, ch->handle_var, 0, op->predicate);
  }

  Stmt Split(Stmt stmt, const ProducerConsumer* env) {
    stmt = Mutate(stmt);
    if (env) {
      stages_.emplace_back(BuildStage(stmt, env->func));
    } else {
      stmt = RemoveNoOp(stmt);
      CHECK(is_no_op(stmt));
    }
    CHECK_NE(stages_.size(), 0);
    stmt = stages_.back();
    for (size_t i = stages_.size() - 1; i != 0; --i) {
      stmt = Block::make(stages_[i - 1], stmt);
    }
    stmt = MarkChannelAccess(cmap_, fifo_map_).Mutate(stmt);
    return RemoveNoOp(stmt);
  }

 private:
  // Build the stage.
  Stmt BuildStage(Stmt body, NodeRef target) {
    int stage_index = static_cast<int>(stages_.size());
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
            op->node, op->attr_key, op->value, no_op));
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
  // Whether split load into a temp fifo.
  bool split_load_{true};
  // Counter for temp FIFOs.
  size_t temp_fifo_count_{0};
  // fifo map
  std::unordered_map<const Variable*, Channel> fifo_map_;
};

class PipelineSplitter : public IRMutator {
 public:
  explicit PipelineSplitter(bool split_load)
      : split_load_(split_load) {}

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == ir::attr::pipeline_exec_scope) {
      CHECK_LE(env_.size(), 1U);
      const ProducerConsumer* env = nullptr;
      if (env_.size() == 1) {
        std::swap(env_[0], env);
      }
      Stmt body = StageSplitter(split_load_).Split(
          op->body, env);
      if (body.same_as(op->body)) return s;
      return AttrStmt::make(
          op->node, op->attr_key, op->value, body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) {
    env_.push_back(op);
    Stmt ret = IRMutator::Mutate_(op, s);
    if (env_.back() == nullptr) {
      ret = ret.as<ProducerConsumer>()->body;
    }
    env_.pop_back();
    return ret;
  }

 private:
  bool split_load_;
  std::vector<const ProducerConsumer *> env_;
};

Stmt SplitPipeline(Stmt stmt, bool split_load) {
  return PipelineSplitter(split_load).Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
