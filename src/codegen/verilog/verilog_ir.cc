/*!
 *  Copyright (c) 2017 by Contributors
 * \file verilog_ir.cc
 */
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <utility>
#include "verilog_ir.h"
#include "../../arithmetic/compute_expr.h"

namespace tvm {
namespace codegen {
namespace verilog {

using namespace ir;

ControlSignal ControlSignalNode::make(
    ControlSignalType type, int advance_size) {
  auto n = make_node<ControlSignalNode>();
  n->ctrl_type = type;
  n->advance_size = advance_size;
  return ControlSignal(n);
}

StageInput StageInputNode::make(Var var, StageInputType input_type) {
  NodePtr<StageInputNode> n = make_node<StageInputNode>();
  n->var = var;
  n->input_type = input_type;
  return StageInput(n);
}

// Replace stage inputs by placeholder, update the input map.
class StageInputReplacer : public IRMutator {
 public:
  explicit StageInputReplacer(
      const std::unordered_map<const Variable*, StageInput>& var_info)
      : var_info_(var_info) {}

  Expr Mutate_(const Variable* op, const Expr& e) final {
    if (replace_.count(op)) {
      return replace_.at(op);
    }
    auto it = var_info_.find(op);
    if (it == var_info_.end()) return e;
    Var new_var(it->second->var->name_hint + ".sync", op->type);
    inputs_.Set(new_var, it->second);
    replace_[op] = new_var;
    return new_var;
  }
  Expr Mutate_(const Load* op, const Expr& e) final {
    CHECK(is_zero(op->index))
        << "Load should be in its own stage.";
    if (replace_.count(op->buffer_var.get())) {
      return replace_.at(op->buffer_var.get());
    }
    auto it = var_info_.find(op->buffer_var.get());
    CHECK(it != var_info_.end())
        << "Load from unknown channel";
    Var data(it->second->var->name_hint + ".load.sync", op->type);
    inputs_.Set(data, it->second);
    replace_[op->buffer_var.get()] = data;
    return data;
  }
  // inputs that get replaced.
  Map<Var, StageInput> inputs_;
  // replacement map
  std::unordered_map<const Variable*, Var> replace_;
  // Variable replacement plan.
  const std::unordered_map<const Variable*, StageInput>& var_info_;
};

/*! \brief Extract module block */
class PipelineExtractor: public IRVisitor {
 public:
  Pipeline Extract(LoweredFunc f) {
    // Initialize the memory map channels
    // TODO(tqchen) move the logic to explicit specification.
    for (auto arg : f->args) {
      if (arg.type().is_handle()) {
        arg_handle_[arg.get()] = arg;
      }
    }
    pipeline_ = make_node<PipelineNode>();
    this->Visit(f->body);
    // setup channels
    for (const auto &kv : cmap_) {
      pipeline_->channels.Set(
          kv.second.node->channel->handle_var,
          ChannelBlock(kv.second.node));
    }
    pipeline_->args = f->args;
    return Pipeline(pipeline_);
  }

  void Visit_(const AttrStmt* op) final {
    if (op->attr_key == attr::pipeline_stage_scope) {
      CHECK(!in_pipeline_stage_);
      in_pipeline_stage_ = true;
      trigger_.emplace_back(std::make_pair(loop_.size(), op));
      IRVisitor::Visit_(op);
      trigger_.pop_back();
      in_pipeline_stage_ = false;
    } else if (op->attr_key == attr::channel_read_advance ||
               op->attr_key == attr::channel_write_advance) {
      trigger_.emplace_back(std::make_pair(loop_.size(), op));
      IRVisitor::Visit_(op);
      trigger_.pop_back();
    } else if (op->attr_key == attr::channel_read_scope ||
               op->attr_key == attr::channel_write_scope) {
      Channel ch(op->node.node_);
      ChannelEntry& cb = cmap_[ch->handle_var.get()];
      if (cb.node != nullptr) {
        CHECK(cb.node->channel.same_as(ch));
      } else {
        cb.node = make_node<ChannelBlockNode>();
        cb.node->channel = ch;
      }
      if (op->attr_key == attr::channel_read_scope) {
        CHECK_EQ(cb.read_ref_count, 0)
            << "One channel can only be read from one consumer";
        ++cb.read_ref_count;
        CHECK(arith::GetConstInt(op->value, &(cb.node->read_window)))
              << "Only supprt constant read window";
      } else {
        CHECK_EQ(cb.write_ref_count, 0)
            << "One channel can only be write by one producer";
        ++cb.write_ref_count;
        CHECK(arith::GetConstInt(op->value, &(cb.node->write_window)))
              << "Only supprt constant write window";
      }
      var_info_[ch->handle_var.get()] =
          StageInputNode::make(ch->handle_var, kChannel);
      IRVisitor::Visit_(op);
      var_info_.erase(ch->handle_var.get());
    } else {
      IRVisitor::Visit_(op);
    }
  }
  void Visit_(const Block* op) final {
    CHECK(!in_pipeline_stage_)
        << "Do not support serial execution inside pipeline";
    IRVisitor::Visit_(op);
  }
  void Visit_(const IfThenElse* op) final {
    LOG(FATAL) << "Not implemeneted";
  }
  void Visit_(const For* op) final {
    if (in_pipeline_stage_) {
      loop_.push_back(
          For::make(op->loop_var, op->min, op->extent,
                    op->for_type, op->device_api, Evaluate::make(0)));
      var_info_[op->loop_var.get()] =
          StageInputNode::make(Var(op->loop_var.node_), kLoopVar);
      IRVisitor::Visit_(op);
      var_info_.erase(op->loop_var.get());
      loop_.pop_back();
    } else {
      IRVisitor::Visit_(op);
    }
  }
  void Visit_(const Store* op) final {
    // Check the access pattern
    Channel arg_write =
        CheckArgHandleAccess(op->buffer_var.get(), op->value.type(), false);
    this->Visit(op->value);
    // The replace logic
    StageInputReplacer repl(var_info_);
    // Setup the compute block.
    NodePtr<ComputeBlockNode> compute =
        make_node<ComputeBlockNode>();
    compute->loop = Array<Stmt>(loop_);
    // setup the advance triggers
    for (const auto& e : trigger_) {
      const AttrStmt* attr = e.second;
      Channel ch;
      if (attr->attr_key == attr::pipeline_stage_scope) {
        ch = arg_write;
        if (!ch.defined()) continue;
      } else {
        ch = Channel(attr->node.node_);
      }
      NodePtr<SignalTriggerNode> trigger
          = make_node<SignalTriggerNode>();
      trigger->channel_var = ch->handle_var;
      // predicate for the trigger
      Expr predicate = const_true();
      for (size_t i = e.first; i < loop_.size(); ++i) {
        const For* loop = loop_[i].as<For>();
        predicate = predicate &&
            (loop->loop_var == (loop->extent - 1));
      }
      trigger->predicate = ir::Simplify(predicate);
      // Add the signal back to the channels.
      ChannelEntry& cb = cmap_.at(ch->handle_var.get());
      trigger->signal_index = static_cast<int>(cb.node->ctrl_signals.size());
      // Grab the advance constant size.
      int trigger_size = 0;
      if (attr->attr_key == attr::pipeline_stage_scope) {
        cb.node->ctrl_signals.push_back(
            ControlSignalNode::make(kComputeFinish, 0));
      } else if (attr->attr_key == attr::channel_read_advance) {
        CHECK(arith::GetConstInt(attr->value, &trigger_size))
            << "Only support constant advance size";
        cb.node->ctrl_signals.push_back(
            ControlSignalNode::make(kReadAdvance, trigger_size));
      } else {
        CHECK(arith::GetConstInt(attr->value, &trigger_size))
            << "Only support constant advance size";
        cb.node->ctrl_signals.push_back(
            ControlSignalNode::make(kWriteAdvance, trigger_size));
      }
      compute->triggers.push_back(SignalTrigger(trigger));
    }

    // Check if we are writing to FIFO.
    const Load* load = op->value.as<Load>();
    if (is_zero(op->index) && load) {
      compute->body = Store::make(
          op->buffer_var,
          Load::make(load->type, load->buffer_var,
                     repl.Mutate(load->index), op->predicate),
          op->index, op->predicate);
    } else {
      compute->body = Store::make(
          op->buffer_var, repl.Mutate(op->value),
          repl.Mutate(op->index), op->predicate);
    }
    compute->inputs = repl.inputs_;
    pipeline_->stages.push_back(ComputeBlock(compute));
  }
  void Visit_(const LetStmt* op) final {
    LOG(FATAL) << "cannot pass through let";
  }
  void Visit_(const Evaluate* op) final {
    LOG(FATAL) << "Not implemeneted";
  }
  void Visit_(const Allocate* op) final {
    CHECK(!in_pipeline_stage_);
  }
  void Visit_(const AssertStmt* op) final {
    LOG(FATAL) << "Not implemeneted";
  }
  void Visit_(const Load* op) final {
    CheckArgHandleAccess(op->buffer_var.get(), op->type, true);
  }
  Channel CheckArgHandleAccess(const Variable* var, Type dtype, bool read_access) {
    if (!arg_handle_.count(var)) return Channel();
    CHECK(!cmap_.count(var))
        << "Multiple access to the same handle";
    ChannelEntry& cb = cmap_[var];
    cb.node = make_node<ChannelBlockNode>();
    cb.node->channel = ChannelNode::make(arg_handle_.at(var), dtype);
    return cb.node->channel;
  }

 private:
  // The channel information.
  struct ChannelEntry {
    NodePtr<ChannelBlockNode> node;
    int read_ref_count{0};
    int write_ref_count{0};
  };
  // Whether we are inside the pipeline stage.
  bool in_pipeline_stage_{false};
  // The current loop nest
  std::vector<Stmt> loop_;
  // Advance signal trigger
  std::vector<std::pair<size_t, const AttrStmt*> > trigger_;
  // Read write scope
  std::vector<const AttrStmt*> channel_scope_;
  // The loop index.
  std::unordered_map<const Variable*, StageInput> var_info_;
  // The channel entry;
  std::unordered_map<const Variable*, ChannelEntry> cmap_;
  // The argument handle map
  std::unordered_map<const Variable*, Var> arg_handle_;
  // The result block.
  NodePtr<PipelineNode> pipeline_;
};

Pipeline MakePipeline(LoweredFunc f) {
  return PipelineExtractor().Extract(f);
}
}  // namespace verilog
}  // namespace codegen
}  // namespace tvm
