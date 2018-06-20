/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_ops.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule_pass.h>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "./graph.h"
#include "../op/op_util.h"
#include "../pass/ir_util.h"

namespace tvm {
namespace schedule {

using namespace ir;

Stmt MakePipeline(const Stage& s,
                  const std::unordered_map<IterVar, Range>& dom_map,
                  Stmt consumer,
                  bool debug_keep_trivial_loop) {
  Stmt producer = s->op->BuildProvide(s, dom_map, debug_keep_trivial_loop);
  if (producer.defined()) {
    producer = ProducerConsumer::make(s->op, true, producer);
  }
  if (s->double_buffer) {
    producer = AttrStmt::make(
        s->op, ir::attr::double_buffer_scope, 1, producer);
  }
  Stmt pipeline = producer;

  if (consumer.defined() && !is_no_op(consumer)) {
    consumer = ProducerConsumer::make(s->op, false, consumer);
    pipeline = Block::make(producer, consumer);
  }
  pipeline = s->op->BuildRealize(s, dom_map, pipeline);
  // use attribute to mark scope of the operation.
  pipeline = AttrStmt::make(
      s->op, ir::attr::realize_scope,
      StringImm::make(s->scope),
      pipeline);

  if (s->is_opengl) {
    pipeline = AttrStmt::make(
        s->op, ir::attr::opengl_stage_scope, StringImm::make(""), pipeline);
  }
  return pipeline;
}

// inject the operator's realization on the stmt.
class InjectAttach : public IRMutator {
 public:
  InjectAttach(const Stage& stage,
               const Stage& attach_spec,
               const std::unordered_map<IterVar, Range>& dom_map,
               bool debug_keep_trivial_loop)
      : stage_(stage), attach_spec_(attach_spec), dom_map_(dom_map),
        debug_keep_trivial_loop_(debug_keep_trivial_loop) {}

  Stmt Mutate(Stmt stmt) final {
    CHECK(stmt.defined());
    stmt =  IRMutator::Mutate(stmt);
    const AttrStmt* op = stmt.as<AttrStmt>();
    if (op != nullptr &&
        op->attr_key == attr::loop_scope) {
      if (attach_spec_->attach_type == kScope &&
          op->node == attach_spec_->attach_ivar) {
        CHECK(!found_attach)
            << "Find IterVar" << attach_spec_->attach_ivar
            << " in multiple places in the IR";
        found_attach = true;
        stmt = AttrStmt::make(
            op->node, op->attr_key, op->value,
            MakePipeline(stage_, dom_map_, op->body, debug_keep_trivial_loop_));
      }
    }
    return stmt;
  }
  // whether attach point is found
  bool found_attach{false};

 private:
  // The stage.
  const Stage& stage_;
  // The attach spec, may not contain op.
  const Stage& attach_spec_;
  // domain map
  const std::unordered_map<IterVar, Range>& dom_map_;
  // Whether keep trivial loops with extent of 1 during lowering.
  // This is a debug feature for dataflow/axis analysis
  bool debug_keep_trivial_loop_;
};

// inject the operator's realization on the stmt.
class InjectScanStep : public IRMutator {
 public:
  InjectScanStep(const Stage& stage,
                 const Operation& scan_op,
                 const std::unordered_map<IterVar, Range>& dom_map,
                 bool is_init,
                 bool debug_keep_trivial_loop)
      : stage_(stage), scan_op_(scan_op),
        dom_map_(dom_map), is_init_(is_init), debug_keep_trivial_loop_(debug_keep_trivial_loop) {}

  Stmt Mutate(Stmt stmt) final {
    CHECK(stmt.defined());
    stmt =  IRMutator::Mutate(stmt);
    // update
    const AttrStmt* op = stmt.as<AttrStmt>();
    if (op != nullptr &&
        ((op->attr_key == attr::scan_update_scope && !is_init_) ||
         (op->attr_key == attr::scan_init_scope && is_init_))) {
      if (op->node.same_as(scan_op_)) {
        found_attach = true;
        stmt = AttrStmt::make(
            op->node, op->attr_key, op->value,
            MakePipeline(stage_, dom_map_, op->body, debug_keep_trivial_loop_));
      }
    }
    return stmt;
  }

  // whether attach point is found
  bool found_attach{false};

 private:
  // the operations to be carried
  const Stage& stage_;
  const Operation& scan_op_;
  // domain map
  const std::unordered_map<IterVar, Range>& dom_map_;
  // whether it is init.
  bool is_init_;
  // Whether keep trivial loops with extent of 1 during lowering.
  // This is a debug feature for dataflow/axis analysis
  bool debug_keep_trivial_loop_;
};

// Postprocessing of schedule op
// Replace the init and update's expression by scan's buffer.
class SchedulePostProc : public IRMutator {
 public:
  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) final {
    auto it = replace_op_.find(op->func.get());
    if (it != replace_op_.end()) {
      Stmt body = this->Mutate(op->body);
      if (it->second.defined()) {
        return ProducerConsumer::make(
            it->second, op->is_producer, body);
      } else {
        return body;
      }
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
  Stmt Mutate_(const LetStmt* op, const Stmt& s) final {
    if (!HasSideEffect(op->value)) {
      var_value_[op->var.get()] = Mutate(op->value);
      return this->Mutate(op->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::loop_scope ||
        op->attr_key == attr::scan_init_scope) {
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::scan_update_scope) {
      const ScanOpNode* scan = op->node.as<ScanOpNode>();
      CHECK(scan);
      var_value_[scan->scan_axis->var.get()] = op->value;
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::thread_extent) {
      // delete duplicated thread extent attr
      auto it = thread_extent_scope_.find(op->node.get());
      if (it != thread_extent_scope_.end()) {
        CHECK(is_zero(ir::Simplify(it->second - op->value)));
        return this->Mutate(op->body);
      } else {
        thread_extent_scope_[op->node.get()] = op->value;
        Stmt ret = IRMutator::Mutate_(op, s);
        thread_extent_scope_.erase(op->node.get());
        return ret;
      }
    } else if (op->attr_key == ir::attr::realize_scope ||
               op->attr_key == ir::attr::double_buffer_scope) {
      auto it = replace_op_.find(op->node.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          Stmt ret = AttrStmt::make(
              it->second, op->attr_key, op->value, op->body);
          return this->Mutate(ret);
        } else {
          return this->Mutate(op->body);
        }
      }
    } else if (op->attr_key == ir::attr::buffer_bind_scope) {
      Array<NodeRef> tuple(op->node.node_);
      Tensor tensor(tuple[1].node_);
      auto it = replace_op_.find(tensor->op.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          return AttrStmt::make(
              Array<NodeRef>{tuple[0], it->second.output(tensor->value_index)},
              op->attr_key, op->value, Mutate(op->body));
        } else {
          return this->Mutate(op->body);
        }
      }
    } else if (op->attr_key == ir::attr::buffer_dim_align) {
      Tensor tensor(op->node.node_);
      auto it = replace_op_.find(tensor->op.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          return AttrStmt::make(
              it->second.output(tensor->value_index),
              op->attr_key, op->value, Mutate(op->body));
        } else {
          return this->Mutate(op->body);
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize* op, const Stmt& s) final {
    TensorKey key{op->func, op->value_index};
    auto it = replace_realize_.find(key);
    if (it != replace_realize_.end()) {
      if (it->second.defined()) {
        Stmt ret = Realize::make(
            it->second->op, it->second->value_index,
            op->type, op->bounds, op->condition, op->body);
        return this->Mutate(ret);
      } else {
        return this->Mutate(op->body);
      }
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const Provide* op, const Stmt& s) final {
    TensorKey key{op->func, op->value_index};
    auto it = replace_buffer_.find(key);
    if (it != replace_buffer_.end()) {
      const Tensor& dst = it->second;
      Stmt ret = Provide::make(
          dst->op, dst->value_index, op->value, op->args);
      return this->Mutate(ret);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Expr Mutate_(const Call* op, const Expr& e) final {
    if (op->call_type == Call::Halide) {
      TensorKey key{op->func, op->value_index};
      auto it = replace_buffer_.find(key);
      if (it != replace_buffer_.end()) {
        const Tensor& dst = it->second;
        Expr ret = Call::make(
            op->type, dst->op->name, op->args,
            op->call_type, dst->op, dst->value_index);
        return this->Mutate(ret);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = var_value_.find(op);
    if (it != var_value_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

  void Init(const Schedule& sch) {
    for (Stage s : sch->stages) {
      for (auto kv : s->iter_var_attrs) {
        // Update bind thread information.
        if (kv.second->bind_thread.defined()) {
          const Var& from = kv.first->var;
          const Var& to = kv.second->bind_thread->var;
          CHECK(!var_value_.count(from.get()));
          var_value_[from.get()] = to;
        }
      }
      // This must be checked for all ops, including scan.
      if (!s->op.same_as(s->origin_op)) {
        for (int i = 0; i < s->op->num_outputs(); ++i) {
          Tensor target = s->origin_op.output(i);
          AddReplace(s->op.output(i), target,
                     target, s->origin_op);
        }
      }
      // Specially add replacements for scan op.
      if (s->op.as<ScanOpNode>()) {
        const ScanOpNode* scan = s->op.as<ScanOpNode>();
        for (size_t i = 0; i < scan->update.size(); ++i) {
          Tensor t = s->origin_op.output(i);
          AddReplace(scan->init[i], t);
          AddReplace(scan->update[i], t);
          AddReplace(scan->state_placeholder[i], t);
        }
      }
    }
  }

 private:
  void AddReplace(Tensor src,
                  Tensor dst,
                  Tensor repl_realize = Tensor(),
                  Operation repl_op = Operation()) {
    TensorKey key{src->op, src->value_index};
    replace_buffer_[key] = dst;
    replace_realize_[key] = repl_realize;
    replace_op_[src->op.get()] = repl_op;
  }
  // The thread extent scope.
  std::unordered_map<const Node*, Expr> thread_extent_scope_;
  // The scan value
  std::unordered_map<const Variable*, Expr> var_value_;
  // buffer replacement
  std::unordered_map<TensorKey, Tensor> replace_buffer_;
  // buffere realization to be replaced
  std::unordered_map<TensorKey, Tensor> replace_realize_;
  // replace producer consumer.
  std::unordered_map<const Node*, Operation> replace_op_;
};

Stmt ScheduleOps(
    Schedule sch, Map<IterVar, Range> dom_map_, bool debug_keep_trivial_loop) {
  Stmt body = Stmt();
  std::unordered_map<IterVar, Range> dom_map = as_unordered_map(dom_map_);
  // scan init and scan updates
  std::unordered_map<Operation, Operation> scan_init;
  for (Stage s : sch->stages) {
    const ScanOpNode* scan = s->op.as<ScanOpNode>();
    if (!scan) continue;
    for (Tensor t : scan->init) {
      if (scan_init.count(t->op)) {
        CHECK(scan_init.at(t->op).same_as(s->op))
            << "Scan init tensor can only belong to one scan";
      } else {
        scan_init[t->op] = s->op;
      }
    }
  }
  // verify correctness of group.
  for (Stage g : sch->groups) {
    CHECK(!g->op.defined());
    CHECK_EQ(g->leaf_iter_vars.size(), 0U);
  }
  // reverse the post DFS order.
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage s = sch->stages[i - 1];
    CHECK_NE(s->attach_type, kInline)
        << "call schedule.normalize before scheduleops";
    CHECK(s->op.defined());
    // no need to specify place holder op.
    if (s->op.as<PlaceholderOpNode>()) continue;
    // Remove grouping sugar, get the real attach spec.
    Stage attach_spec = s.GetAttachSpec();

    if (scan_init.count(s->op)) {
      CHECK(body.defined());
      InjectScanStep mu(s, scan_init.at(s->op), dom_map, true, debug_keep_trivial_loop);
      body = mu.Mutate(body);
      CHECK(mu.found_attach)
          << "did not find attachment point for scan.init";
    } else if (attach_spec->attach_type == kScanUpdate) {
      // Handle scan update
      CHECK(body.defined());
      InjectScanStep mu(s, attach_spec->attach_stage->op, dom_map, false, debug_keep_trivial_loop);
      body = mu.Mutate(body);
      CHECK(mu.found_attach)
          << "did not find attachment point for scan.update";
    } else if (attach_spec->attach_type == kInlinedAlready) {
      // do nothing
    } else if (attach_spec->attach_type == kGroupRoot) {
      CHECK(!s->group.defined());
      body = MakePipeline(s, dom_map, body, debug_keep_trivial_loop);
    } else {
      CHECK_EQ(attach_spec->attach_type, kScope);
      CHECK(body.defined());
      InjectAttach mutator(s, attach_spec, dom_map, debug_keep_trivial_loop);
      body = mutator.Mutate(body);
      CHECK(mutator.found_attach)
          << "did not find attachment point for " << s << " in "
          << attach_spec->attach_stage->op  << " x " << attach_spec->attach_ivar
          << ", body:\n"
          << body;
    }
  }
  SchedulePostProc post_proc;
  post_proc.Init(sch);
  return post_proc.Mutate(body);
}

}  // namespace schedule
}  // namespace tvm
