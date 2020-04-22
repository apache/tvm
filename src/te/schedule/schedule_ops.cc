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
 * \file schedule_ops.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "graph.h"
#include "../operation/op_util.h"
#include "../../tir/transforms/ir_util.h"

namespace tvm {
namespace te {

using namespace tir;

Stmt MakePipeline(const Stage& s,
                  const std::unordered_map<IterVar, Range>& dom_map,
                  Stmt consumer,
                  bool debug_keep_trivial_loop) {
  Stmt producer = s->op->BuildProvide(s, dom_map, debug_keep_trivial_loop);
  if (s->double_buffer) {
    producer = AttrStmtNode::make(
        s->op, tir::attr::double_buffer_scope, 1, producer);
  }
  Stmt pipeline = producer;

  if (consumer.defined() && !is_no_op(consumer)) {
    pipeline = SeqStmt({producer, consumer});
  }
  pipeline = s->op->BuildRealize(s, dom_map, pipeline);
  // use attribute to mark scope of the operation.
  pipeline = AttrStmtNode::make(
      s->op, tir::attr::realize_scope,
      StringImmNode::make(s->scope),
      pipeline);

  if (s->is_opengl) {
    pipeline = AttrStmtNode::make(
        s->op, tir::attr::opengl_stage_scope, StringImmNode::make(""), pipeline);
  }
  return pipeline;
}

// inject the operator's realization on the stmt.
class InjectAttach : public StmtMutator {
 public:
  InjectAttach(const Stage& stage,
               const Stage& attach_spec,
               const std::unordered_map<IterVar, Range>& dom_map,
               bool debug_keep_trivial_loop)
      : stage_(stage), attach_spec_(attach_spec), dom_map_(dom_map),
        debug_keep_trivial_loop_(debug_keep_trivial_loop) {}

  Stmt VisitStmt(const Stmt& input_stmt) final {
    CHECK(input_stmt.defined());
    auto stmt = StmtMutator::VisitStmt(input_stmt);
    const AttrStmtNode* op = stmt.as<AttrStmtNode>();
    if (op != nullptr &&
        op->attr_key == tir::attr::loop_scope) {
      if (attach_spec_->attach_type == kScope &&
          op->node == attach_spec_->attach_ivar) {
        CHECK(!found_attach)
            << "Find IterVar" << attach_spec_->attach_ivar
            << " in multiple places in the IR";
        found_attach = true;
        stmt = AttrStmtNode::make(
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
class InjectScanStep : public StmtMutator {
 public:
  InjectScanStep(const Stage& stage,
                 const Operation& scan_op,
                 const std::unordered_map<IterVar, Range>& dom_map,
                 bool is_init,
                 bool debug_keep_trivial_loop)
      : stage_(stage), scan_op_(scan_op),
        dom_map_(dom_map), is_init_(is_init), debug_keep_trivial_loop_(debug_keep_trivial_loop) {}

  Stmt VisitStmt(const Stmt& input_stmt) final {
    CHECK(input_stmt.defined());
    auto stmt = StmtMutator::VisitStmt(input_stmt);
    // update
    const AttrStmtNode* op = stmt.as<AttrStmtNode>();
    if (op != nullptr &&
        ((op->attr_key == tir::attr::scan_update_scope && !is_init_) ||
         (op->attr_key == tir::attr::scan_init_scope && is_init_))) {
      if (op->node.same_as(scan_op_)) {
        found_attach = true;
        stmt = AttrStmtNode::make(
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
class SchedulePostProc : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const LetStmtNode* op) final {
    if (!HasSideEffect(op->value)) {
      var_value_[op->var.get()] = this->VisitExpr(op->value);
      return this->VisitStmt(op->body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::loop_scope ||
        op->attr_key == tir::attr::scan_init_scope) {
      return this->VisitStmt(op->body);
    } else if (op->attr_key == tir::attr::scan_update_scope) {
      const ScanOpNode* scan = op->node.as<ScanOpNode>();
      CHECK(scan);
      var_value_[scan->scan_axis->var.get()] = op->value;
      return this->VisitStmt(op->body);
    } else if (op->attr_key == tir::attr::thread_extent) {
      // delete duplicated thread extent attr
      auto it = thread_extent_scope_.find(op->node.get());
      if (it != thread_extent_scope_.end()) {
        CHECK(is_zero(analyzer_.Simplify(it->second - op->value)));
        return this->VisitStmt(op->body);
      } else {
        thread_extent_scope_[op->node.get()] = op->value;
        Stmt ret = StmtExprMutator::VisitStmt_(op);
        thread_extent_scope_.erase(op->node.get());
        return ret;
      }
    } else if (op->attr_key == tir::attr::realize_scope ||
               op->attr_key == tir::attr::double_buffer_scope) {
      auto it = replace_op_.find(op->node.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          Stmt ret = AttrStmtNode::make(
              it->second, op->attr_key, op->value, op->body);
          return this->VisitStmt(ret);
        } else {
          return this->VisitStmt(op->body);
        }
      }
    } else if (op->attr_key == tir::attr::buffer_bind_scope) {
      Array<ObjectRef> tuple = Downcast<Array<ObjectRef> >(op->node);
      Tensor tensor = Downcast<Tensor>(tuple[1]);
      auto it = replace_op_.find(tensor->op.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          return AttrStmtNode::make(
              Array<ObjectRef>{tuple[0], it->second.output(tensor->value_index)},
              op->attr_key, op->value, this->VisitStmt(op->body));
        } else {
          return this->VisitStmt(op->body);
        }
      }
    } else if (op->attr_key == tir::attr::buffer_dim_align) {
      Tensor tensor = Downcast<Tensor>(op->node);
      auto it = replace_op_.find(tensor->op.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          return AttrStmtNode::make(
              it->second.output(tensor->value_index),
              op->attr_key, op->value, this->VisitStmt(op->body));
        } else {
          return this->VisitStmt(op->body);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const RealizeNode* op) final {
    TensorKey key{op->func, op->value_index};
    auto it = replace_realize_.find(key);
    if (it != replace_realize_.end()) {
      if (it->second.defined()) {
        Stmt ret = RealizeNode::make(
            it->second->op, it->second->value_index,
            op->dtype, op->bounds, op->condition, op->body);
        return this->VisitStmt(ret);
      } else {
        return this->VisitStmt(op->body);
      }
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const ProvideNode* op) final {
    TensorKey key{op->func, op->value_index};
    auto it = replace_buffer_.find(key);
    if (it != replace_buffer_.end()) {
      const Tensor& dst = it->second;
      Stmt ret = ProvideNode::make(
          dst->op, dst->value_index, op->value, op->args);
      return this->VisitStmt(ret);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->call_type == CallNode::Halide) {
      TensorKey key{op->func, op->value_index};
      auto it = replace_buffer_.find(key);
      if (it != replace_buffer_.end()) {
        const Tensor& dst = it->second;
        PrimExpr ret = CallNode::make(
            op->dtype, dst->op->name, op->args,
            op->call_type, dst->op, dst->value_index);
        return this->VisitExpr(ret);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_value_.find(op);
    if (it != var_value_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
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
      if (const ScanOpNode* scan = s->op.as<ScanOpNode>()) {
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
  std::unordered_map<const Object*, PrimExpr> thread_extent_scope_;
  // The scan value
  std::unordered_map<const VarNode*, PrimExpr> var_value_;
  // buffer replacement
  std::unordered_map<TensorKey, Tensor> replace_buffer_;
  // buffere realization to be replaced
  std::unordered_map<TensorKey, Tensor> replace_realize_;
  // replace producer consumer.
  std::unordered_map<const Object*, Operation> replace_op_;
  // integer analyzer
  arith::Analyzer analyzer_;
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
      body = mu(std::move(body));
      CHECK(mu.found_attach)
          << "did not find attachment point for scan.init";
    } else if (attach_spec->attach_type == kScanUpdate) {
      // Handle scan update
      CHECK(body.defined());
      InjectScanStep mu(s, attach_spec->attach_stage->op, dom_map, false, debug_keep_trivial_loop);
      body = mu(std::move(body));
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
      body = mutator(std::move(body));
      CHECK(mutator.found_attach)
          << "did not find attachment point for " << s << " in "
          << attach_spec->attach_stage->op  << " x " << attach_spec->attach_ivar
          << ", body:\n"
          << body;
    }
  }
  SchedulePostProc post_proc;
  post_proc.Init(sch);
  return post_proc(std::move(body));
}

TVM_REGISTER_GLOBAL("schedule.ScheduleOps")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args.size() == 2)
    *ret = ScheduleOps(args[0], args[1], false);
  else
    *ret = ScheduleOps(args[0], args[1], args[2]);
});

}  // namespace te
}  // namespace tvm
