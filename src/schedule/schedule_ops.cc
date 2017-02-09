/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_ops.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/schedule_pass.h>

#include "../pass/ir_util.h"
#include "../arithmetic/compute_expr.h"
#include "./graph.h"

namespace tvm {
namespace schedule {

using namespace arith;
using namespace ir;

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
    } else {
      // Always restrict threaded IterVar to starts from 0.
      CHECK(is_zero(dom->min));
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt::make(iv, "thread_extent", dom->extent, no_op));
      value_map[iv] = var;
    }
    if (!reduce_init_loop) {
      // annotate the extent of the IterVar
      nest[i + 1].emplace_back(
          AttrStmt::make(iv, "scope", iv->var, no_op));
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
  auto nest = MakeLoopNest(s, dom_map, 0, false,
                           bound_state, {}, &value_map);

  provide = Substitute(provide, value_map);
  if (init.defined()) {
    // try to find the location to insert the initialization.
    // Fuse the initialization and provide loop when possible.
    std::unordered_map<IterVar, int> reduce_state;
    const ComputeOpNode* compute = s->op.as<ComputeOpNode>();
    for (IterVar iv : compute->reduce_axis) {
      reduce_state[iv] = 2;
    }
    for (IterVar iv : compute->axis) {
      reduce_state[iv] = 1;
    }
    // find which iter var is related to reduction and which is related to axis.
    PassDownFlag(s, &reduce_state);
    auto leaf_iter_vars = s->leaf_iter_vars;
    std::unordered_map<IterVar, Expr> init_value_map;
    // first first loop that is related to reduction.
    size_t begin_loop = leaf_iter_vars.size();
    for (size_t i = 0; i < leaf_iter_vars.size(); ++i) {
      auto iv = leaf_iter_vars[i];
      int flag = reduce_state.at(iv);
      if ((flag & 2) != 0) {
        begin_loop = i; break;
      }
      init_value_map[iv] = value_map.at(iv);
    }
    // skip loops that does not relates to axis.
    std::unordered_map<IterVar, bool> skip_iter;
    for (auto kv : reduce_state) {
      int flag = kv.second;
      if ((flag & 1) == 0) skip_iter[kv.first] = true;
    }
    auto init_nest = MakeLoopNest(
        s, dom_map, begin_loop, true,
        bound_state, skip_iter, &init_value_map);
    init = Substitute(init, init_value_map);
    init  = MergeNest(init_nest, init);
    // common nest
    std::vector<std::vector<Stmt> > common(nest.begin(), nest.begin() + begin_loop);
    std::vector<std::vector<Stmt> > reduce(nest.begin() + begin_loop, nest.end());
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
  if (compute) {
    if (compute->reduce_axis.size() == 0) {
      provide = MakeProvide(compute, tensors);
    } else {
      MakeReduction(compute, tensors, &init, &provide);
    }
  } else {
    LOG(FATAL) << "not supported op " << s->op->type_key();
  }

  Stmt producer = MakeLoop(s, dom_map, provide, init);
  producer = ProducerConsumer::make(s->op, true, producer);

  Stmt pipeline = producer;
  if (consumer.defined()) {
    consumer = ProducerConsumer::make(s->op, false, consumer);
    pipeline = Block::make(producer, consumer);
  }

  if (s->op.as<ComputeOpNode>()) {
    pipeline = MakeRealize(s->op.as<ComputeOpNode>(),
                           dom_map, tensors, pipeline);
  } else {
    LOG(FATAL) << "not supported op";
    return Stmt();
  }
  // use attribute to mark scope of the operation.
  pipeline = AttrStmt::make(
      s->op, "realize_scope",
      StringImm::make(s->scope),
      pipeline);
  return pipeline;
}

// inject the operator's realization on the stmt.
class InjectRealize : public IRMutator {
 public:
  InjectRealize(Stage schedule, Map<IterVar, Range> dom_map)
      : schedule(schedule), dom_map(dom_map) {}

  Stmt Mutate(Stmt stmt) final {
    CHECK(stmt.defined());
    stmt =  IRMutator::Mutate(stmt);
    const AttrStmt* op = stmt.as<AttrStmt>();
    if (op != nullptr &&
        op->type_key == "scope") {
      if (op->node == schedule->attach_ivar) {
        CHECK(!found_attach);
        found_attach = true;
        stmt = AttrStmt::make(
            op->node, op->type_key, op->value,
            MakePipeline(schedule, dom_map,
                         IRMutator::Mutate(op->body)));
      }
    }
    return stmt;
  }
  // the operations to be carried
  Stage schedule;
  // domain map
  Map<IterVar, Range> dom_map;
  // whether attach point is found
  bool found_attach{false};
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

Stmt ScheduleOps(
    Schedule sch, Map<IterVar, Range> dom_map) {
  Stmt body = Stmt();
  // reverse the post DFS order.
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage s = sch->stages[i - 1];
    // no need to specify place holder op.
    if (s->op.as<PlaceholderOpNode>()) continue;
    if (s->attach_type == kInline) {
      body = InjectInline(s->op, body);
    } else if (s->attach_type == kRoot || s-> attach_type == kNone) {
      body = MakePipeline(s, dom_map, body);
    } else if (s->attach_type == kScope) {
      CHECK(body.defined());
      InjectRealize mutator(s, dom_map);
      body = mutator.Mutate(body);
      CHECK(mutator.found_attach)
          << "did not find attachment point";
    }
  }
  return body;
}

}  // namespace schedule
}  // namespace tvm
