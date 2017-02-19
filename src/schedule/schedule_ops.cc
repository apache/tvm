/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_ops.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/schedule_pass.h>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "../pass/ir_util.h"
#include "../arithmetic/compute_expr.h"
#include "./graph.h"

namespace tvm {
namespace schedule {

using namespace arith;
using namespace ir;

// Two private scope marks
namespace attr {
constexpr const char* loop_scope = "loop_scope";
constexpr const char* scan_update_scope = "scan_update_scope";
constexpr const char* scan_init_scope = "scan_init_scope";
}  // namespace attr

/*!
 * \brief message passing to find if IterVar is related to reduction.
 * \param s The stage to be used.
 * \param p_state The message passing state
 *     IterVar->flag
 */
void PassDownFlag(const Stage& s,
                   std::unordered_map<IterVar, int>* p_state) {
  auto& state = *p_state;
  for (IterVarRelation rel : s->relations) {
    if (rel.as<SplitNode>()) {
      const SplitNode* s = rel.as<SplitNode>();
      int flag = state.at(s->parent);
      state[s->outer] = flag;
      state[s->inner] = flag;
    } else if (rel.as<FuseNode>()) {
      const FuseNode* s = rel.as<FuseNode>();
      int flag_outer = state.at(s->outer);
      int flag_inner = state.at(s->inner);
      state[s->fused] = flag_outer | flag_inner;
    } else if (rel.as<RebaseNode>()) {
      const RebaseNode* s = rel.as<RebaseNode>();
      int flag = state.at(s->parent);
      state[s->rebased] = flag;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

/*!
 * \brief message passing to find if boundary checking on IterVar is needed.
 * \param s The stage to be used.
 * \param p_state The message passing state
 *     IterVar->flag
 */
void PassUpBoundCheck(const Stage& s,
                      const Map<IterVar, Range>& dom_map,
                      std::unordered_map<IterVar, bool>* p_state) {
  auto& state = *p_state;
  using Halide::Internal::can_prove;
  for (size_t i = s->relations.size(); i != 0; --i) {
    IterVarRelation rel = s->relations[i - 1];
    if (rel.as<SplitNode>()) {
      const SplitNode* s = rel.as<SplitNode>();
      bool outer = state.at(s->outer);
      bool inner = state.at(s->inner);
      Expr factor = dom_map.at(s->inner)->extent;
      Expr step = dom_map.at(s->outer)->extent;

      if (outer || inner) {
        state[s->parent] = true;
      } else {
        if (can_prove(dom_map.at(s->parent)->extent == factor * step)) {
          state[s->parent] = false;
        } else {
          state[s->parent] = true;
        }
      }
    } else if (rel.as<FuseNode>()) {
      const FuseNode* s = rel.as<FuseNode>();
      bool fused = state.at(s->fused);
      state[s->outer] = fused;
      state[s->inner] = fused;
    } else if (rel.as<RebaseNode>()) {
      const RebaseNode* s = rel.as<RebaseNode>();
      state[s->parent] = state.at(s->rebased);
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

/*!
 * \brief use message passing to calculate the assignment of each Var inside the loop body.
 * \param s The schedule to be used.
 * \param dom_map The domain map of each iteration variable's domain
 * \param p_state The message passing state
 *     IterVar->The assignment.
 */
void PassUpOffset(const Stage& s,
                  const Map<IterVar, Range>& dom_map,
                  std::unordered_map<IterVar, Expr>* p_state) {
  auto& state = *p_state;
  for (size_t i = s->relations.size(); i != 0; --i) {
    IterVarRelation rel = s->relations[i - 1];
    if (rel.as<SplitNode>()) {
      const SplitNode* s = rel.as<SplitNode>();
      Expr outer = state.at(s->outer);
      Expr inner = state.at(s->inner);
      Expr factor = dom_map.at(s->inner)->extent;
      Expr parent_min = dom_map.at(s->parent)->min;
      state[s->parent] = inner + outer * factor;
      // add min if they exist
      if (!is_zero(parent_min)) {
        state[s->parent] = state[s->parent] + parent_min;
      }
    } else if (rel.as<FuseNode>()) {
      const FuseNode* s = rel.as<FuseNode>();
      Expr value = state.at(s->fused);
      Expr factor = dom_map.at(s->inner)->extent;
      Expr outer_min = dom_map.at(s->outer)->min;
      Expr inner_min = dom_map.at(s->inner)->min;
      state[s->outer] = value / factor;
      state[s->inner] = value % factor;
      // add min if they exist
      if (!is_zero(outer_min)) {
        state[s->outer] = state[s->outer] + outer_min;
      }
      if (!is_zero(inner_min)) {
        state[s->inner] = state[s->inner] + inner_min;
      }
    } else if (rel.as<RebaseNode>()) {
      const RebaseNode* s = rel.as<RebaseNode>();
      Expr value = state.at(s->rebased);
      Expr parent_min = dom_map.at(s->parent)->min;
      // add min if they exist
      if (!is_zero(parent_min)) {
        state[s->parent] = value + parent_min;
      } else {
        state[s->parent] = value;
      }
    } else {
      LOG(FATAL) << "unknown relation type";
    }
  }
}

std::vector<std::vector<Stmt> >
MakeLoopNest(const Stage& sch,
             const Map<IterVar, Range>& dom_map,
             size_t begin_loop,
             bool reduce_init_loop,
             const std::unordered_map<IterVar, bool>& bound_state,
             const std::unordered_map<IterVar, bool>& skip_iter,
             std::unordered_map<IterVar, Expr>* p_value_map) {
  auto leaf_iter_vars = sch->leaf_iter_vars;
  Stmt no_op = Evaluate::make(0);
  // create the loop nest
  std::vector<std::vector<Stmt> > nest;
  nest.resize(leaf_iter_vars.size() + 1);
  std::unordered_map<IterVar, Expr>& value_map = *p_value_map;

  for (size_t i = begin_loop; i < leaf_iter_vars.size(); ++i) {
    auto iv = leaf_iter_vars[i];
    if (skip_iter.count(iv) && skip_iter.at(iv)) {
      // skip this iteration.
      value_map[iv] = iv->var;
      continue;
    }
    Range dom = dom_map.at(iv);
    // initialize the offset and loop_level
    Var var = iv->var;
    if (reduce_init_loop) {
      var = Var(iv->var->name_hint + ".init", iv->var.type());
    }
    // Mark the iter var in the IR, to remember the point
    if (iv->thread_tag.length() == 0) {
      ForType for_type = ForType::Serial;
      if (sch->iter_var_attrs.count(iv)) {
        switch (sch->iter_var_attrs[iv]->iter_type) {
          case kUnrolled: for_type = ForType::Unrolled; break;
          case kVectorized: for_type = ForType::Vectorized; break;
        }
      }
      if (is_one(dom->extent)) {
        nest[i + 1].emplace_back(
            LetStmt::make(var, dom->min, no_op));
        value_map[iv] = dom->min;
      } else if (is_zero(dom->min)) {
        nest[i + 1].emplace_back(
            For::make(var, 0, dom->extent,
                      for_type, DeviceAPI::None, no_op));
        value_map[iv] = var;
      } else {
        Var idx(iv->var->name_hint + ".idx", iv->var.type());
        nest[i + 1].emplace_back(
            For::make(idx, 0, dom->extent,
                      for_type, DeviceAPI::None, no_op));
        Expr new_value = dom->min + idx;
        value_map[iv] = new_value;
        nest[i + 1].emplace_back(
            LetStmt::make(var, new_value, no_op));
      }
    } else if (iv->thread_tag == "vthread") {
      // virtual thread
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      CHECK(is_positive_const(dom->extent));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt::make(iv, ir::attr::virtual_thread, dom->extent, no_op));
      value_map[iv] = var;
    } else {
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt::make(iv, ir::attr::thread_extent, dom->extent, no_op));
      value_map[iv] = var;
    }
    if (!reduce_init_loop) {
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt::make(iv, attr::loop_scope, iv->var, no_op));
    }
  }
  // message passing to get offset of root iter vars.
  PassUpOffset(sch, dom_map, &value_map);

  // insert conditions
  for (IterVar iv : sch->op->root_iter_vars()) {
    if (skip_iter.count(iv)) continue;
    Range dom = dom_map.at(iv);
    if (bound_state.at(iv)) {
      Expr condition = ComputeExpr<Sub>(value_map.at(iv), dom->min) < dom->extent;
      nest.back().emplace_back(IfThenElse::make(condition, no_op));
    }
    CHECK(iv->dom.defined());
    if (!reduce_init_loop && !iv->dom.same_as(dom)) {
      Expr condition = ComputeExpr<Sub>(value_map.at(iv), iv->dom->min) < iv->dom->extent;
      nest.back().emplace_back(IfThenElse::make(condition, no_op));
    }
  }
  return nest;
}

Stmt Substitute(Stmt s,
                const std::unordered_map<IterVar, Expr>& value_map) {
  Map<Var, Expr> temp;
  for (const auto& kv : value_map) {
    temp.Set(kv.first->var, kv.second);
  }
  return ir::Substitute(s, temp);
}

Stmt MakeLoop(const Stage& s,
              const Map<IterVar, Range>& dom_map,
              Stmt provide,
              Stmt init) {
  std::unordered_map<IterVar, Expr> value_map;
  // bound check state.
  std::unordered_map<IterVar, bool> bound_state;
  for (IterVar iv : s->leaf_iter_vars) {
    bound_state[iv] = false;
  }
  PassUpBoundCheck(s, dom_map, &bound_state);
  auto nest = MakeLoopNest(
      s, dom_map, 0, false,
      bound_state, {{}}, &value_map);

  provide = Substitute(provide, value_map);
  if (init.defined()) {
    // try to find the location to insert the initialization.
    // Fuse the initialization and provide loop when possible.
    std::unordered_map<IterVar, int> update_state;
    const ComputeOpNode* compute = s->op.as<ComputeOpNode>();
    const ScanOpNode* scan = s->op.as<ScanOpNode>();
    if (compute) {
      for (IterVar iv : compute->reduce_axis) {
        update_state[iv] = 2;
      }
      for (IterVar iv : compute->axis) {
        update_state[iv] = 1;
      }
    } else if (scan) {
      update_state[scan->scan_axis] = 2;
      for (IterVar iv : s->outermost_threads) {
        update_state[iv] = 1;
      }
    }
    // find which iter var is related to reduction and which is related to axis.
    PassDownFlag(s, &update_state);
    auto leaf_iter_vars = s->leaf_iter_vars;
    std::unordered_map<IterVar, Expr> init_value_map;
    // first first loop that is related to reduction.
    size_t begin_loop = leaf_iter_vars.size();
    for (size_t i = 0; i < leaf_iter_vars.size(); ++i) {
      auto iv = leaf_iter_vars[i];
      int flag = update_state.at(iv);
      if ((flag & 2) != 0) {
        begin_loop = i; break;
      }
      init_value_map[iv] = value_map.at(iv);
    }
    // skip loops that does not relates to axis.
    std::unordered_map<IterVar, bool> skip_iter;
    for (auto kv : update_state) {
      int flag = kv.second;
      if ((flag & 1) == 0) skip_iter[kv.first] = true;
    }
    auto init_nest = MakeLoopNest(
        s, dom_map, begin_loop, true,
        bound_state, skip_iter, &init_value_map);
    init = Substitute(init, init_value_map);
    init  = MergeNest(init_nest, init);
    // common nest
    std::vector<std::vector<Stmt> > common(nest.begin(), nest.begin() + begin_loop + 1);
    std::vector<std::vector<Stmt> > reduce(nest.begin() + begin_loop + 1, nest.end());
    provide = MergeNest(reduce, provide);
    return MergeNest(
        common, Block::make(init, provide));
  } else {
    return MergeNest(nest, provide);
  }
}

Stmt MakeProvide(const ComputeOpNode* op,
                 const std::vector<Tensor>& tensors) {
  Tensor t = tensors[0];
  Array<Expr> args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  return Provide::make(t->op, t->value_index, op->body, args);
}

Stmt MakeRealize(const ComputeOpNode* op,
                 const Map<IterVar, Range>& dom_map,
                 const std::vector<Tensor>& tensors,
                 Stmt body) {
  Tensor t = tensors[0];
  Halide::Internal::Region bounds;
  for (IterVar iv : op->axis) {
    bounds.push_back(dom_map.at(iv));
  }
  return Realize::make(t->op, t->value_index, t->dtype,
                       bounds, make_const(Bool(1), true), body);
}

Stmt MakeRealize(const ScanOpNode* op,
                 const Map<IterVar, Range>& dom_map,
                 const std::vector<Tensor>& tensors,
                 Stmt body) {
  Range sdom = dom_map.at(op->scan_axis);
  Range tdom = Range::make_with_min_extent(
      0, ir::Simplify(sdom->extent + sdom->min));
  size_t sp_idx = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const Tensor& t = tensors[i];
    CHECK_EQ(static_cast<size_t>(t->value_index), i);
    Halide::Internal::Region bounds;
    bounds.push_back(tdom);
    for (size_t k = 1; k < op->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = op->spatial_axis_[sp_idx];
      bounds.push_back(dom_map.at(sp_ax));
    }
    body = Realize::make(t->op, t->value_index, t->dtype,
                         bounds, make_const(Bool(1), true), body);
  }
  return body;
}


void MakeReduction(const ComputeOpNode* op,
                   const std::vector<Tensor>& tensors,
                   Stmt* init,
                   Stmt* provide) {
  Stmt no_op = Evaluate::make(0);
  Tensor t = tensors[0];
  std::vector<Stmt> nest;
  Array<Expr>  args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  const Reduce* reduce = op->body.as<Reduce>();
  CHECK(reduce);
  Expr init_value, update_value;
  if (reduce->op == "Add") {
    init_value = make_zero(reduce->type);
    update_value = Add::make(t(args), reduce->source);
  } else if (reduce->op == "Max") {
    init_value = reduce->type.min();
    update_value = Max::make(t(args), reduce->source);
  } else if (reduce->op == "Min") {
    init_value = reduce->type.max();
    update_value = Min::make(t(args), reduce->source);
  } else {
    LOG(FATAL) << "Unsupported reduction " << reduce->op;
  }
  *init = Provide::make(t->op, t->value_index, init_value, args);
  *provide = Provide::make(t->op, t->value_index, update_value, args);
}

Stmt MakePipeline(const Stage& s,
                  const Map<IterVar, Range>& dom_map,
                  Stmt consumer) {
  std::vector<Tensor> tensors;
  for (int i = 0; i < s->op->num_outputs(); ++i) {
    tensors.emplace_back(s->op.output(i));
  }

  Stmt init, provide;

  const ComputeOpNode* compute = s->op.as<ComputeOpNode>();
  const ScanOpNode* scan = s->op.as<ScanOpNode>();
  if (compute) {
    if (compute->reduce_axis.size() == 0) {
      provide = MakeProvide(compute, tensors);
    } else {
      MakeReduction(compute, tensors, &init, &provide);
    }
  } else if (scan) {
    // Provide is done by the sub operations.
    provide = AttrStmt::make(
        s->op, attr::scan_update_scope, scan->scan_axis->var,
        Evaluate::make(0));
    init = AttrStmt::make(
        s->op, attr::scan_init_scope, 0,
        Evaluate::make(0));
  } else {
    LOG(FATAL) << "not supported op " << s->op->type_key();
  }

  Stmt producer = MakeLoop(s, dom_map, provide, init);
  producer = ProducerConsumer::make(s->op, true, producer);

  Stmt pipeline = producer;
  // check if consumer is nop.
  bool is_no_op{false};
  const Evaluate* ev = consumer.as<Evaluate>();
  if (ev && ev->value.as<IntImm>()) is_no_op = true;

  if (consumer.defined() && !is_no_op) {
    consumer = ProducerConsumer::make(s->op, false, consumer);
    pipeline = Block::make(producer, consumer);
  }

  if (s->op.as<ComputeOpNode>()) {
    pipeline = MakeRealize(s->op.as<ComputeOpNode>(),
                           dom_map, tensors, pipeline);
  } else if (s->op.as<ScanOpNode>()) {
    pipeline = MakeRealize(s->op.as<ScanOpNode>(),
                           dom_map, tensors, pipeline);
  } else {
    LOG(FATAL) << "not supported op";
  }
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
                const Map<IterVar, Range>& dom_map)
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
  const Map<IterVar, Range>& dom_map_;
  // internal stack about realization scope.
  std::vector<const Node*> producer_;
};

// inject the operator's realization on the stmt.
class InjectScanStep : public IRMutator {
 public:
  InjectScanStep(const Stage& stage,
                 const Operation& scan_op,
                 const Map<IterVar, Range>& dom_map,
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
  const Map<IterVar, Range>& dom_map_;
  // whether it is init.
  bool is_init_;
};

Stmt InjectInline(const Operation op, Stmt body) {
  CHECK(body.defined());

  const ComputeOpNode* compute = op.as<ComputeOpNode>();
  CHECK(compute != nullptr)
      << "can only inline compute op";
  Array<Var> args;
  for (auto iv : compute->axis) {
    args.push_back(iv->var);
  }
  return Inline(body, op, args, compute->body);
}

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
      if (s->op.as<ScanOpNode>()) {
        const ScanOpNode* scan = s->op.as<ScanOpNode>();
        for (size_t i = 0; i < scan->update.size(); ++i) {
          Tensor t = s->origin_op.output(i);
          AddReplace(scan->init[i], t);
          AddReplace(scan->update[i], t);
          AddReplace(scan->state_placeholder[i], t);
        }
      } else if (!s->op.same_as(s->origin_op)) {
        Tensor target = s->origin_op.output(0);
        AddReplace(s->op.output(0), target,
                   target, s->origin_op);
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
    Schedule sch, Map<IterVar, Range> dom_map) {
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
