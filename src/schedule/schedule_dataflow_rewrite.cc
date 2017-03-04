/*!
 *  Copyright (c) 2017 by Contributors
 * \file schedule_dataflow_rewrite.cc
 */
#include <tvm/schedule.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_set>

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

using ir::TensorKey;

// The replacer of cache.
class TensorReplacer : public ir::IRMutator {
 public:
  explicit TensorReplacer(const std::unordered_map<TensorKey, Tensor>& vmap)
      : vmap_(vmap) {}
  Expr Mutate_(const ir::Call* op, const Expr& e) {
    if (op->call_type == ir::Call::Halide) {
      ir::TensorKey key{op->func, op->value_index};
      auto it = vmap_.find(key);
      if (it != vmap_.end()) {
        Expr ret = ir::Call::make(
            op->type, it->second->op->name, op->args,
            op->call_type, it->second->op, it->second->value_index);
        found = true;
        return IRMutator::Mutate_(ret.as<ir::Call>(), ret);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  // whether it is found.
  bool found{false};

 private:
  const std::unordered_map<TensorKey, Tensor>& vmap_;
};

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
                     std::unordered_map<TensorKey, Tensor>* vmap) {
  for (Stage s : stages) {
    if (s->op.as<ComputeOpNode>()) {
      const ComputeOpNode* compute = s->op.as<ComputeOpNode>();
      TensorReplacer repl(*vmap);
      Expr body = repl.Mutate(compute->body);
      if (repl.found) {
        Operation op = ComputeOpNode::make(
            compute->name, compute->axis, body);
        (*vmap)[TensorKey{s->op, 0}] = op.output(0);
        s->op = op;
      }
    } else if (s->op.as<ScanOpNode>()) {
      const ScanOpNode* scan = s->op.as<ScanOpNode>();
      std::shared_ptr<ScanOpNode> n =
          std::make_shared<ScanOpNode>(*scan);
      // copy on write semantics ganrantees correctness
      for (size_t i = 0; i < n->init.size(); ++i) {
        TensorKey key{n->init[i]->op, n->init[i]->value_index};
        if (vmap->count(key)) {
          n->init.Set(i, vmap->at(key));
        }
      }
      for (size_t i = 0; i < n->update.size(); ++i) {
        TensorKey key{n->update[i]->op, n->update[i]->value_index};
        if (vmap->count(key)) {
          n->update.Set(i, vmap->at(key));
        }
      }
      if (!n->init.same_as(scan->init) ||
          !n->update.same_as(scan->update)) {
        Operation op(n);
        for (int i = 0; i < op->num_outputs(); ++i) {
          (*vmap)[TensorKey{s->op, i}] = op.output(i);
        }
        s->op = op;
      }
    } else if (s->op.as<PlaceholderOpNode>()) {
    } else {
      LOG(FATAL) << "unhandled problem";
    }
  }
}

Tensor Schedule::cache_read(const Tensor& tensor,
                            const std::string& scope,
                            const Array<Operation>& readers) {
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
  std::unordered_map<TensorKey, Tensor> vsub;
  vsub[TensorKey{tensor->op, tensor->value_index}] = cache;

  std::unordered_map<TensorKey, Tensor> vmap;
  for (Operation op : readers) {
    const ComputeOpNode* compute = op.as<ComputeOpNode>();
    CHECK(compute)
        << "cache read only take ComputeOp as readers";
    Stage s = operator[](op);
    compute = s->op.as<ComputeOpNode>();

    TensorReplacer repl(vsub);
    Expr body = repl.Mutate(compute->body);
    CHECK(repl.found)
        << "Cannot find " << tensor
        << " in the body of specified reader " << op;
    Operation repl_op = ComputeOpNode::make(
        compute->name, compute->axis, body);
    vmap[TensorKey{s->op, 0}] = repl_op.output(0);
    s->op = repl_op;
  }
  ReplaceDataFlow((*this)->stages, &vmap);
  ArrayNode* stages = (*this)->stages.CopyOnWrite();
  size_t pos = FindNodeRef(stages, operator[](tensor->op));
  Stage cache_stage = Stage(cache->op);
  cache_stage.set_scope(scope);
  CHECK_LT(pos, stages->data.size());
  stages->data.insert(stages->data.begin() + pos + 1,
                      cache_stage.node_);
  (*this)->stage_map.Set(cache->op, cache_stage);
  return cache;
}

Tensor Schedule::cache_write(const Tensor& tensor,
                             const std::string& scope) {
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

  std::unordered_map<TensorKey, Tensor> vmap;
  vmap[TensorKey{orig_stage->op, 0}] = orig_new_op.output(0);
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
  return cache_tensor;
}


void RebaseNonZeroMinLoop(const Schedule& sch) {
  std::unordered_map<IterVar, IterVar> rebase_map;
  std::unordered_map<const Node*, int> attach_mark;

  for (Stage s : sch->stages) {
    if (s->attach_type == kScope) {
      attach_mark[s->attach_stage.get()] = 1;
    }
    if (s->op.as<ScanOpNode>()) {
      attach_mark[s.get()] = 1;
    }
  }

  for (Stage s : sch->stages) {
    if (!attach_mark.count(s.get())) continue;
    auto root_iter_vars = s->op->root_iter_vars();
    ArrayNode* leaf_vars = s->leaf_iter_vars.CopyOnWrite();
    for (IterVar iv : root_iter_vars) {
      size_t idx = FindNodeRef(leaf_vars, iv);
      if (idx < leaf_vars->data.size()) {
        // insert rebase
        IterVar rebased = IterVarNode::make(
            Range(), iv->var.copy_with_suffix(".rb"), iv->iter_type);
        s->relations.push_back(RebaseNode::make(iv, rebased));
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
}

void SetScanAttach(const Schedule& sch) {  // NOLINT(*)
  for (Stage stage : sch->stages) {
    if (stage->attach_type == kScanUpdate) {
      const Stage& parent = stage->attach_stage;
      stage->attach_ivar =
          parent->leaf_iter_vars[parent->leaf_iter_vars.size() - 1];
    }
  }
}


void InjectInline(const Schedule& sch) {
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
  std::unordered_map<TensorKey, Tensor> repl;
  // rewrite dataflow
  for (size_t i = 0; i < sch->stages.size(); ++i) {
    if (new_body[i].defined() &&
        !new_body[i].same_as(sch->stages[i]->op)) {
      const ComputeOpNode* compute = sch->stages[i]->op.as<ComputeOpNode>();
      CHECK(compute);
      Operation op = ComputeOpNode::make(
          compute->name, compute->axis, new_body[i]);
      repl[TensorKey{sch->stages[i]->op, 0}] = op.output(0);
      Stage s = sch->stages[i];
      s->op = op;
    }
  }
  ReplaceDataFlow(sch->stages, &repl);
}

void Schedule::normalize() {
  RebaseNonZeroMinLoop(*this);
  SetScanAttach(*this);
  InjectInline(*this);
}

}  // namespace tvm
