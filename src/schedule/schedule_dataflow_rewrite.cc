/*!
 *  Copyright (c) 2017 by Contributors
 * \file schedule_dataflow_rewrite.cc
 */
#include <tvm/schedule.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include "./message_passing.h"

namespace tvm {

// find first occurance location in leaf
template<typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Node* n = v.get();
  for (size_t i = 0; i < array_node->data.size(); ++i) {
    if (array_node->data[i].get() == n) return i;
  }
  return array_node->data.size();
}

// The replacer of cache.
class VarReplacer : public ir::IRMutator {
 public:
  explicit VarReplacer(
      const std::unordered_map<const Variable*, Expr>& vsub)
      : vsub_(vsub) {}
  Expr Mutate_(const Variable* op, const Expr& e) {
    auto it = vsub_.find(op);
    if (it != vsub_.end()) return it->second;
    return e;
  }

 private:
  const std::unordered_map<const Variable*, Expr>& vsub_;
};

// Replace data flow appears in all stages given the tensor change.
// Also update vmap if subsequent dataflow need to be replaced.
void ReplaceDataFlow(const Array<Stage>& stages,
                     std::unordered_map<Tensor, Tensor>* vmap) {
  for (Stage s : stages) {
    Operation op = s->op->ReplaceInputs(s->op, *vmap);
    if (!op.same_as(s->op)) {
      for (int i = 0; i < op->num_outputs(); ++i) {
        (*vmap)[s->op.output(i)] = op.output(i);
      }
      s->op = op;
    }
  }
}

Tensor Schedule::cache_read(const Tensor& tensor,
                            const std::string& scope,
                            const Array<Operation>& readers) {
  (*this)->InvalidateCache();
  // create identity mapping.
  std::ostringstream os;
  os << tensor->op->name;
  if (tensor->op->num_outputs() != 1) {
    os << ".v" << tensor->value_index;
  }
  os << "." << scope;

  Tensor cache = compute(tensor->shape, [&tensor](const Array<Var>& i) {
      return tensor(Array<Expr>(i.begin(), i.end()));
    }, os.str());
  std::unordered_map<Tensor, Tensor> vsub;
  vsub[tensor] = cache;

  std::unordered_map<Tensor, Tensor> vmap;
  for (Operation op : readers) {
    Stage s = operator[](op);
    Operation repl_op = s->op->ReplaceInputs(s->op, vsub);
    CHECK(!repl_op.same_as(s->op))
        << "Cannot find " << tensor
        << " in the inputs of " << s->op;
    vmap[s->op.output(0)] = repl_op.output(0);
    s->op = repl_op;
  }
  ReplaceDataFlow((*this)->stages, &vmap);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  Stage op_stage = operator[](tensor->op);
  size_t pos = FindNodeRef(stages, op_stage);
  Stage cache_stage = Stage(cache->op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos + 1,
                      cache_stage.node_);
  (*this)->stage_map.Set(cache->op, cache_stage);
  // Update group
  cache_stage->group = op_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }
  return cache;
}

Tensor Schedule::cache_write(const Tensor& tensor,
                             const std::string& scope) {
  (*this)->InvalidateCache();
  Stage orig_stage = operator[](tensor->op);
  const ComputeOpNode* compute = tensor->op.as<ComputeOpNode>();
  CHECK(compute)
      << "cache write only take ComputeOp as writers";
  CHECK_EQ(orig_stage->relations.size(), 0U)
      << "Create cache_write before doing split/fuse/reorder";
  compute = orig_stage->op.as<ComputeOpNode>();
  CHECK(compute);
  Array<Expr> args;
  Array<IterVar> new_axis;
  std::unordered_map<const Variable*, Expr> vsub;
  for (IterVar iv : compute->axis) {
    args.push_back(iv->var);
    IterVar new_iv = IterVarNode::make(
        iv->dom, iv->var.copy_with_suffix(".c"), iv->iter_type);
    new_axis.push_back(new_iv);
    vsub[iv->var.get()] = new_iv->var;
  }
  VarReplacer repl(vsub);
  Expr body = repl.Mutate(compute->body);
  Operation cache_op = ComputeOpNode::make(
      compute->name + "." + scope, new_axis, body);
  Tensor cache_tensor = cache_op.output(0);
  Operation orig_new_op = ComputeOpNode::make(
      compute->name, compute->axis,
      cache_tensor(args));

  std::unordered_map<Tensor, Tensor> vmap;
  vmap[orig_stage->op.output(0)] = orig_new_op.output(0);
  ReplaceDataFlow((*this)->stages, &vmap);
  // mutate orig stage
  orig_stage->op = orig_new_op;
  orig_stage->all_iter_vars = orig_stage->op->root_iter_vars();
  orig_stage->leaf_iter_vars = orig_stage->all_iter_vars;
  // create schedule for new cached stage.
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  size_t pos = FindNodeRef(stages, orig_stage);
  Stage cache_stage = Stage(cache_op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos,
                      cache_stage.node_);
  (*this)->stage_map.Set(cache_op, cache_stage);
  // Update group
  cache_stage->group = orig_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }
  return cache_tensor;
}

void RebaseNonZeroMinLoop(const Schedule& sch) {
  std::unordered_map<IterVar, IterVar> rebase_map;
  for (Stage s : sch->stages) {
    if (s->attach_type == kInlinedAlready) continue;

    auto root_iter_vars = s->op->root_iter_vars();
    ArrayNode* leaf_vars = s->leaf_iter_vars.CopyOnWrite();
    for (IterVar iv : root_iter_vars) {
      size_t idx = FindNodeRef(leaf_vars, iv);
      auto it  = s->iter_var_attrs.find(iv);
      // don;t need to rebase path that are binded.
      if (it != s->iter_var_attrs.end() &&
          (*it).second->bind_thread.defined()) {
        continue;
      }
      if (idx < leaf_vars->data.size()) {
        // insert rebase
        IterVar rebased = IterVarNode::make(
            Range(), iv->var.copy_with_suffix(""), iv->iter_type);
        s->relations.push_back(RebaseNode::make(iv, rebased));
        if (s->iter_var_attrs.count(iv)) {
          s->iter_var_attrs.Set(rebased, s->iter_var_attrs.at(iv));
        }
        leaf_vars->data[idx] = rebased.node_;
        rebase_map[iv] = rebased;
      }
    }
  }
  // remap the parent relation
  for (Stage s : sch->stages) {
    if (s->attach_type != kScope) continue;
    if (rebase_map.count(s->attach_ivar)) {
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
  for (Stage s : sch->groups) {
    if (s->attach_type != kScope) continue;
    if (rebase_map.count(s->attach_ivar)) {
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
}

void InjectInline(ScheduleNode* sch) {
  sch->InvalidateCache();
  std::vector<Expr> new_body(sch->stages.size());
  // inline all the ops
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage stage = sch->stages[i - 1];
    if (stage->attach_type == kInline) {
      stage->attach_type = kInlinedAlready;
      Array<Var> args;
      Expr body;
      {
        // setup args
        const ComputeOpNode* compute = stage->op.as<ComputeOpNode>();
        CHECK(compute)
            << "can only inline compute op";
        for (auto iv : compute->axis) {
          args.push_back(iv->var);
        }
        body = compute->body;
      }
      for (size_t j = i; j < sch->stages.size(); ++j) {
        Stage s = sch->stages[j];
        const ComputeOpNode* compute = s->op.as<ComputeOpNode>();
        if (compute) {
          if (!new_body[j].defined()) {
            new_body[j] = s->op.as<ComputeOpNode>()->body;
          }
          new_body[j] = ir::Inline(ir::Evaluate::make(new_body[j]),
                                   stage->op, args, body).as<ir::Evaluate>()->value;
        }
      }
    }
  }
  std::unordered_map<Tensor, Tensor> repl;
  // rewrite dataflow
  for (size_t i = 0; i < sch->stages.size(); ++i) {
    if (new_body[i].defined() &&
        !new_body[i].same_as(sch->stages[i]->op)) {
      const ComputeOpNode* compute = sch->stages[i]->op.as<ComputeOpNode>();
      CHECK(compute);
      Operation op = ComputeOpNode::make(
          compute->name, compute->axis, new_body[i]);
      repl[sch->stages[i]->op.output(0)] = op.output(0);
      Stage s = sch->stages[i];
      s->op = op;
    }
  }
  ReplaceDataFlow(sch->stages, &repl);
}

Schedule Schedule::normalize() {
  Schedule sn = copy();
  InjectInline(sn.operator->());
  RebaseNonZeroMinLoop(sn);
  return sn;
}

// Handle reduction factor.
Tensor Schedule::rfactor(const Tensor& tensor,
                         const IterVar& axis) {
  (*this)->InvalidateCache();
  using ir::Reduce;
  CHECK_EQ(axis->iter_type, kCommReduce)
      << "Can only factor reduction axis";
  Stage reduce_stage = operator[](tensor->op);
  const ComputeOpNode* compute_op = reduce_stage->op.as<ComputeOpNode>();
  CHECK(compute_op) << "Can only factor  ComputeOp";
  ArrayNode* leaf_vars = reduce_stage->leaf_iter_vars.CopyOnWrite();
  {
    size_t axis_pos = FindNodeRef(leaf_vars, axis);
    CHECK_NE(axis_pos, leaf_vars->data.size())
        << "Cannot find IterVar " << axis << " in leaf iter vars";
  }
  // Find touched reduction axis.
  std::unordered_map<IterVar, int> touch_map;
  touch_map[axis] = 1;
  schedule::PassUpBitMaskOr(reduce_stage, &touch_map, true);
  schedule::PassDownBitMaskOr(reduce_stage, &touch_map, true);
  // Verify normal axis are not touched.
  for (IterVar iv : compute_op->axis) {
    CHECK(!touch_map.count(iv))
        << "Factor axis touches normal axis.";
  }
  // Get the replace index
  std::unordered_map<IterVar, Range> dom_map;
  std::unordered_map<IterVar, Expr> value_map;
  for (IterVar iv : compute_op->reduce_axis) {
    if (touch_map.count(iv)) dom_map[iv] = iv->dom;
  }
  schedule::PassDownDomain(reduce_stage, &dom_map, true);
  for (IterVar iv : reduce_stage->leaf_iter_vars) {
    if (touch_map.count(iv)) {
      Range dom = dom_map.at(iv);
      if (is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = iv->var;
      }
    }
  }
  schedule::PassUpIndex(reduce_stage, dom_map, &value_map, true);
  // Get the factored op node.
  auto n = std::make_shared<ComputeOpNode>();
  n->name = compute_op->name + ".rf";
  {
    // axis relacement.
    auto iv_node = std::make_shared<IterVarNode>();
    iv_node->dom = dom_map.at(axis);
    CHECK(is_zero(iv_node->dom->min))
        << "Can only factor reduction domain starting from 0";
    iv_node->var = axis->var;
    iv_node->iter_type = kDataPar;
    n->axis.push_back(IterVar(iv_node));

    for (IterVar iv : compute_op->axis) {
      n->axis.push_back(iv);
    }
  }
  // predicate generation, copy not touched axis.
  const Reduce* reduce = compute_op->body.as<Reduce>();
  CHECK(reduce) << "Can only rfactor non-inline reductions";
  Expr predicate = reduce->condition;
  std::unordered_map<const Variable*, Expr> vsub;
  for (IterVar iv : compute_op->reduce_axis) {
    if (!touch_map.count(iv)) {
      n->reduce_axis.push_back(iv);
    } else {
      CHECK(value_map.count(iv));
      Expr index = value_map.at(iv);
      vsub[iv->var.get()] = index;
      if (!index.same_as(iv->var)) {
        Expr cond = (index < dom_map.at(iv)->extent);
        if (is_one(predicate)) {
          predicate = cond;
        } else {
          predicate = predicate && cond;
        }
      }
    }
  }
  // Copy touched axis.
  for (IterVar iv : reduce_stage->leaf_iter_vars) {
    if (touch_map.count(iv) && !iv.same_as(axis)) {
      CHECK_EQ(iv->iter_type, kCommReduce);
      auto ncpy = std::make_shared<IterVarNode>(*iv.operator->());
      ncpy->dom = dom_map.at(iv);
      n->reduce_axis.push_back(IterVar(ncpy));
    }
  }
  n->body = Reduce::make(reduce->combiner,
                         VarReplacer(vsub).Mutate(reduce->source),
                         n->reduce_axis,
                         predicate);
  // refresh relations, keep the un-touched relations.
  Array<IterVarRelation> rels;
  for (IterVarRelation rel : reduce_stage->relations) {
    bool touched = false;
    if (const SplitNode* r = rel.as<SplitNode>()) {
      if (touch_map.count(r->parent)) touched = true;
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      if (touch_map.count(r->fused)) touched = true;
    } else if (const RebaseNode* r = rel.as<RebaseNode>()) {
      if (touch_map.count(r->parent)) touched = true;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
    if (!touched) {
      rels.push_back(rel);
    }
  }
  // initialize the factored stage.
  Operation factor_op(n);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  size_t stage_pos = FindNodeRef(stages, reduce_stage);
  Stage factor_stage = Stage(factor_op);
  factor_stage->relations = rels;
  CHECK_LT(stage_pos, stages->data.size());
  stages->data.insert(stages->data.begin() + stage_pos,
                      factor_stage.node_);
  (*this)->stage_map.Set(factor_op, factor_stage);
  factor_stage->group = reduce_stage->group;
  if (factor_stage->group.defined()) {
    ++factor_stage->group->num_child_stages;
  }
  // Replace the old reduction.
  IterVar repl_red_axis = reduce_axis(
      dom_map.at(axis), axis->var->name_hint + ".v");
  Tensor factor_tensor = factor_op.output(0);
  Tensor old_tensor = reduce_stage->op.output(0);
  Tensor repl_tensor = compute(old_tensor->shape, [&](const Array<Var>& i) {
      Array<Expr> indices;
      indices.push_back(repl_red_axis->var);
      for (Var v : i) {
        indices.push_back(v);
      }
      return Reduce::make(reduce->combiner,
        factor_tensor(indices), {repl_red_axis}, const_true());
    }, old_tensor->op->name + ".repl");

  std::unordered_map<Tensor, Tensor> vmap;
  vmap[old_tensor] = repl_tensor;
  ReplaceDataFlow((*this)->stages, &vmap);
  // revamp the reduction stage.
  reduce_stage->op = repl_tensor->op;
  reduce_stage->all_iter_vars = repl_tensor->op->root_iter_vars();
  reduce_stage->leaf_iter_vars = reduce_stage->all_iter_vars;
  reduce_stage->relations = Array<IterVarRelation>();
  return factor_tensor;
}
}  // namespace tvm
