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

namespace tvm {
namespace schedule {

using namespace ir;

Stmt MakePipeline(const Stage& s,
                  const std::unordered_map<IterVar, Range>& dom_map,
                  Stmt consumer) {
  Stmt producer = s->op->BuildProvide(s, dom_map);
  if (producer.defined()) {
    producer = ProducerConsumer::make(s->op, true, producer);
  }
  Stmt pipeline = producer;

  if (consumer.defined() && !is_no_op(consumer)) {
    consumer = ProducerConsumer::make(s->op, false, consumer);
    pipeline = Block::make(producer, consumer);
  }
  pipeline = s->op->BuildRealize(s->op, dom_map, pipeline);
  // use attribute to mark scope of the operation.
  pipeline = AttrStmt::make(
      s->op, ir::attr::realize_scope,
      StringImm::make(s->scope),
      pipeline);
  return pipeline;
}

// inject the operator's realization on the stmt.
class InjectAttach : public IRMutator {
 public:
  InjectAttach(const Stage& stage,
               const std::unordered_map<IterVar, Range>& dom_map)
      : stage_(stage), dom_map_(dom_map) {}

  Stmt Mutate(Stmt stmt) final {
    CHECK(stmt.defined());
    stmt =  IRMutator::Mutate(stmt);
    const AttrStmt* op = stmt.as<AttrStmt>();
    if (op != nullptr &&
        op->type_key == attr::loop_scope) {
      CHECK_NE(producer_.size(), 0U);
      if (op->node == stage_->attach_ivar &&
          producer_.back() == stage_->attach_stage->op.get()) {
        CHECK(!found_attach);
        found_attach = true;
        stmt = AttrStmt::make(
            op->node, op->type_key, op->value,
            MakePipeline(stage_, dom_map_, op->body));
      }
    }
    return stmt;
  }
  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) final {
    if (op->is_producer) {
      producer_.push_back(op->func.get());
      Stmt ret = IRMutator::Mutate_(op, s);
      producer_.pop_back();
      return ret;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
  // whether attach point is found
  bool found_attach{false};

 private:
  // the operations to be carried
  const Stage& stage_;
  // domain map
  const std::unordered_map<IterVar, Range>& dom_map_;
  // internal stack about realization scope.
  std::vector<const Node*> producer_;
};

// inject the operator's realization on the stmt.
class InjectScanStep : public IRMutator {
 public:
  InjectScanStep(const Stage& stage,
                 const Operation& scan_op,
                 const std::unordered_map<IterVar, Range>& dom_map,
                 bool is_init)
      : stage_(stage), scan_op_(scan_op),
        dom_map_(dom_map), is_init_(is_init) {}

  Stmt Mutate(Stmt stmt) final {
    CHECK(stmt.defined());
    stmt =  IRMutator::Mutate(stmt);
    // update
    const AttrStmt* op = stmt.as<AttrStmt>();
    if (op != nullptr &&
        ((op->type_key == attr::scan_update_scope && !is_init_) ||
         (op->type_key == attr::scan_init_scope && is_init_))) {
      if (op->node.same_as(scan_op_)) {
        found_attach = true;
        stmt = AttrStmt::make(
            op->node, op->type_key, op->value,
            MakePipeline(stage_, dom_map_, op->body));
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
    if (op->type_key == attr::loop_scope) {
      return this->Mutate(op->body);
    } else if (op->type_key == attr::scan_init_scope) {
      return this->Mutate(op->body);
    } else if (op->type_key == attr::scan_update_scope) {
      const ScanOpNode* scan = op->node.as<ScanOpNode>();
      CHECK(scan);
      var_value_[scan->scan_axis->var.get()] = op->value;
      return this->Mutate(op->body);
    } else if (op->type_key == ir::attr::realize_scope) {
      auto it = replace_op_.find(op->node.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          Stmt ret = AttrStmt::make(
              it->second, op->type_key, op->value, op->body);
          return this->Mutate(ret);
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
      // This must be checked for all ops, including scan.
      if (!s->op.same_as(s->origin_op)) {
        Tensor target = s->origin_op.output(0);
        AddReplace(s->op.output(0), target,
                   target, s->origin_op);
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
    Schedule sch, Map<IterVar, Range> dom_map_) {
  Stmt body = Stmt();
  // scan init and scan updates
  std::unordered_map<Operation, std::pair<Operation, bool> > scan_attach;
  for (Stage s : sch->stages) {
    const ScanOpNode* scan = s->op.as<ScanOpNode>();
    if (!scan) continue;
    for (Tensor t : scan->init) {
      if (scan_attach.count(t->op)) {
        CHECK(scan_attach.at(t->op).first.same_as(s->op))
            << "Scan init tensor can only belong to one scan";
      } else {
        scan_attach[t->op] = std::make_pair(s->op, true);
      }
    }
    for (Tensor t : scan->update) {
      if (scan_attach.count(t->op)) {
        CHECK(scan_attach.at(t->op).first.same_as(s->op))
            << "Scan update tensor can only belong to one scan";
      } else {
        scan_attach[t->op] = std::make_pair(s->op, false);
      }
    }
  }
  std::unordered_map<IterVar, Range> dom_map;
  for (auto kv : dom_map_) {
    dom_map[kv.first] = kv.second;
  }

  // reverse the post DFS order.
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage s = sch->stages[i - 1];
    CHECK_NE(s->attach_type, kInline)
        << "call schedule.normalize before scheduleops";
    // no need to specify place holder op.
    if (s->op.as<PlaceholderOpNode>()) continue;
    if (scan_attach.count(s->op)) {
      CHECK(s->attach_type == kNone ||
            s->attach_type == kScanUpdate)
          << "Cannot specify compute_at for scan's init/update";
      CHECK(body.defined());
      const auto& p = scan_attach.at(s->op);
      InjectScanStep mu(s, p.first, dom_map, p.second);
      body = mu.Mutate(body);
      CHECK(mu.found_attach)
          << "did not find attachment point for scan.init/update";
    } else if (s->attach_type == kInlinedAlready) {
      // do nothing
    } else if (s->attach_type == kRoot || s-> attach_type == kNone) {
      body = MakePipeline(s, dom_map, body);
    } else if (s->attach_type == kScope) {
      CHECK(body.defined());
      InjectAttach mutator(s, dom_map);
      body = mutator.Mutate(body);
      CHECK(mutator.found_attach)
          << "did not find attachment point for " << s << " in"
          << s->attach_stage->op << " x "
          << body;
    }
  }
  SchedulePostProc post_proc;
  post_proc.Init(sch);
  return post_proc.Mutate(body);
}

}  // namespace schedule
}  // namespace tvm
