/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule.cc
 */
#include <tvm/schedule.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>
#include "./graph.h"

namespace tvm {

namespace {

// find first occurance location in leaf
template<typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Node* n = v.get();
  for (size_t i = 0; i < array_node->data.size(); ++i) {
    if (array_node->data[i].get() == n) return i;
  }
  return array_node->data.size();
}

size_t FindLeafVar(ArrayNode* all_vars, ArrayNode* leaf_vars, const IterVar& v) {
  size_t pos = FindNodeRef(leaf_vars, v);
  if (pos < leaf_vars->data.size()) return pos;

  if (FindNodeRef(all_vars, v) < all_vars->data.size()) {
    LOG(FATAL) << "Operate on iter var " << v
               << "that has already been splitted";
  } else {
    LOG(FATAL) << "Operate on iter var " << v
               << "that is not part of the schedule";
  }
  return 0;
}

void Split(StageNode* self, IterVar parent,
           IterVar outer, IterVar inner, Expr factor) {
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  size_t pos = FindLeafVar(all_vars, leaf_vars, parent);

  self->relations.push_back(SplitNode::make(parent, outer, inner, factor));
  // add vars to all vars
  all_vars->data.push_back(outer.node_);
  all_vars->data.push_back(inner.node_);
  // replace the position.
  leaf_vars->data.erase(leaf_vars->data.begin() + pos);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos, inner.node_);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos, outer.node_);
}

}  // namespace

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<StageNode>([](const StageNode *op, IRPrinter *p) {
  p->stream << "stage("
            << op->op
            << ")";
});

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IterVarAttrNode>([](const IterVarAttrNode *op, IRPrinter *p) {
    switch (op->iter_type) {
      case kUnrolled: p->stream << "unroll"; break;
      case kVectorized: p->stream << "vectorize"; break;
    }
  });

Stage::Stage(Operation op) {
  auto n = std::make_shared<StageNode>();
  n->op = op;
  n->origin_op = op;
  n->all_iter_vars = op->root_iter_vars();
  n->leaf_iter_vars = n->all_iter_vars;
  node_ = n;
}

Stage& Stage::set_scope(std::string scope) {  // NOLINT(*)
  (*this)->scope = scope;
  return *this;
}

Stage& Stage::compute_at(Stage parent, IterVar scope) {   // NOLINT(*)
  (*this)->attach_type = kScope;
  (*this)->attach_ivar = scope;
  (*this)->attach_stage = parent;
  bool found = false;
  for (size_t i = 0; i < parent->leaf_iter_vars.size(); ++i) {
    if (scope == parent->leaf_iter_vars[i]) {
      found = true; break;
    }
  }
  CHECK(found)
      << "Cannot find the axis in parent's leaf_iter_vars or outermost_threads";
  return *this;
}

Stage& Stage::compute_inline() {   // NOLINT(*)
  (*this)->attach_type = kInline;
  return *this;
}

Stage& Stage::compute_root() {   // NOLINT(*)
  (*this)->attach_type = kRoot;
  return *this;
}

Stage& Stage::split(
    IterVar parent, IterVar* p_outer, IterVar* p_inner, Expr factor) {  // NOLINT(*)
  // place holder for the splitted results.
  IterVar outer(Range(), parent->var->name_hint + ".outer");
  IterVar inner(Range(), parent->var->name_hint + ".inner");
  *p_outer = outer; *p_inner = inner;

  Split(operator->(), parent, outer, inner, factor);
  return *this;
}

Stage& Stage::split(IterVar parent, IterVar outer, IterVar* p_inner, Expr factor) { // NOLINT(*)
  // place holder for the splitted results.
  IterVar inner(Range(), parent->var->name_hint + ".inner");
  *p_inner = inner;
  Split(operator->(), parent, outer, inner, factor);

  return *this;
}

Stage& Stage::fuse(IterVar inner, IterVar outer, IterVar* p_target) {  // NOLINT(*)
  IterVar fused(Range(), outer->var->name_hint + "." + inner->var->name_hint + ".fused");
  *p_target = fused;
  StageNode* self = operator->();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();

  self->relations.push_back(FuseNode::make(inner, outer, fused));
  all_vars->data.push_back(fused.node_);

  size_t pos_inner = FindLeafVar(all_vars, leaf_vars, inner);
  size_t pos_outer = FindLeafVar(all_vars, leaf_vars, outer);
  CHECK_EQ(pos_inner, pos_outer + 1)
      << "Can only fuse iterations that are consecutive between each other";
  leaf_vars->data.erase(leaf_vars->data.begin() + pos_outer,
                        leaf_vars->data.begin() + pos_inner + 1);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos_outer,
                         fused.node_);
  return *this;
}

Stage& Stage::reorder(const Array<IterVar>& order) {  // NOLINT(*)
  StageNode* self = operator->();
  CHECK(!self->op.as<ScanOpNode>())
      << "Cannot reorder axis of scan";
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  std::vector<size_t> pos;

  for (size_t i = 0; i < order.size(); ++i) {
    pos.push_back(FindLeafVar(all_vars, leaf_vars, order[i]));
  }
  std::vector<std::shared_ptr<Node> > temp;
  for (size_t i = 0; i < pos.size(); ++i) {
    temp.emplace_back(leaf_vars->data[pos[i]]);
  }
  std::sort(pos.begin(), pos.end());
  for (size_t i = 0; i < pos.size(); ++i) {
    leaf_vars->data[pos[i]] = temp[i];
  }
  return *this;
}

Stage& Stage::tile(IterVar x_parent, IterVar y_parent,
                   IterVar* p_x_outer, IterVar* p_y_outer,
                   IterVar* p_x_inner, IterVar* p_y_inner,
                   Expr x_factor, Expr y_factor) { // NOLINT(*)
  split(x_parent, p_x_outer, p_x_inner, x_factor);
  split(y_parent, p_y_outer, p_y_inner, y_factor);
  reorder(Array<IterVar>({*p_x_outer, *p_y_outer, *p_x_inner, *p_y_inner}));
  return *this;
}

Stage& Stage::outermost_threads(Array<IterVar> threads) {
  StageNode* self = operator->();
  CHECK(self->op.as<ScanOpNode>())
      << "outermost_threads is only valid for composite ops such as ScanOp";
  CHECK_EQ(self->outermost_threads.size(), 0U)
      << "Already set outermost_threads";
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  std::vector<std::shared_ptr<Node> > temp;
  for (IterVar iv : threads) {
    temp.push_back(iv.node_);
  }
  leaf_vars->data.insert(
      leaf_vars->data.begin(), temp.begin(), temp.end());
  all_vars->data.insert(
      all_vars->data.end(), temp.begin(), temp.end());
  (*this)->outermost_threads = threads;
  return *this;
}

inline void SetAttr(StageNode* self, IterVar var, IterVarAttr attr) {
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, var);
  auto it = self->iter_var_attrs.find(var);
  if (it != self->iter_var_attrs.end()) {
    CHECK_EQ((*it).second->iter_type, attr->iter_type)
        << "IterVar's is already set to "
        << (*it).second << " instead of " << attr;
  } else {
    self->iter_var_attrs.Set(var, attr);
  }
}

Stage& Stage::vectorize(IterVar var) {   // NOLINT(*)
  SetAttr(operator->(), var, IterVarAttr(kVectorized));
  return *this;
}

Stage& Stage::unroll(IterVar var) {   // NOLINT(*)
  SetAttr(operator->(), var, IterVarAttr(kUnrolled));
  return *this;
}

Schedule::Schedule(Array<Operation> ops) {
  auto n = std::make_shared<ScheduleNode>();
  n->outputs = ops;
  auto g = schedule::CreateReadGraph(n->outputs);
  Array<Operation> post_order = schedule::PostDFSOrder(n->outputs, g);
  // output set.
  std::unordered_set<Operation> output_set;
  for (Operation x : ops) {
    output_set.insert(x);
  }
  for (Operation op : post_order) {
    Stage stage(op);
    stage->is_output = output_set.count(op);
    n->stages.push_back(stage);
    n->stage_map.Set(op, stage);
  }
  node_ = std::move(n);
}

Stage Schedule::operator[](const Operation& op) {
  return (*this)->stage_map.at(op);
}

IterVarRelation SplitNode::make(
    IterVar parent, IterVar outer,
    IterVar inner, Expr factor) {
  auto n = std::make_shared<SplitNode>();
  n->parent = parent;
  n->outer = outer;
  n->inner = inner;
  n->factor = factor;
  return IterVarRelation(n);
}

IterVarRelation FuseNode::make(
    IterVar outer, IterVar inner, IterVar fused) {
  auto n = std::make_shared<FuseNode>();
  n->outer = outer;
  n->inner = inner;
  n->fused = fused;
  return IterVarRelation(n);
}

IterVarRelation RebaseNode::make(IterVar parent, IterVar rebased) {
  auto n = std::make_shared<RebaseNode>();
  n->parent = parent;
  n->rebased = rebased;
  return IterVarRelation(n);
}

void Schedule::normalize() {
  std::unordered_map<IterVar, IterVar> rebase_map;
  std::unordered_map<const Node*, int> attach_mark;


  for (Stage s : (*this)->stages) {
    if (s->attach_type == kScope) {
      attach_mark[s->attach_stage.get()] = 1;
    }
  }

  for (Stage s : (*this)->stages) {
    if (!attach_mark.count(s.get())) continue;
    auto root_iter_vars = s->op->root_iter_vars();
    ArrayNode* leaf_vars = s->leaf_iter_vars.CopyOnWrite();

    for (IterVar iv : root_iter_vars) {
      size_t idx = FindNodeRef(leaf_vars, iv);
      if (idx < leaf_vars->data.size()) {
        // insert rebase
        IterVar rebased(Range(), iv->var->name_hint + ".rb");
        s->relations.push_back(RebaseNode::make(iv, rebased));
        leaf_vars->data[idx] = rebased.node_;
        rebase_map[iv] = rebased;
      }
    }
  }
  // remap the parent relation
  for (Stage s : (*this)->stages) {
    if (s->attach_type != kScope) continue;
    if (rebase_map.count(s->attach_ivar)) {
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
}

IterVarAttr::IterVarAttr(IterVarType t) {
  std::shared_ptr<IterVarAttrNode> n = std::make_shared<IterVarAttrNode>();
  n->iter_type = t;
  node_ = n;
}

TVM_REGISTER_NODE_TYPE(StageNode);
TVM_REGISTER_NODE_TYPE(IterVarAttrNode);
TVM_REGISTER_NODE_TYPE(SplitNode);
TVM_REGISTER_NODE_TYPE(FuseNode);
TVM_REGISTER_NODE_TYPE(RebaseNode);
TVM_REGISTER_NODE_TYPE(ScheduleNode);

using ir::TensorKey;

// The replacer of cache.
class TensorReplacer : public ir::IRMutator {
 public:
  TensorReplacer(const std::unordered_map<TensorKey, Tensor>& vmap)
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
        << " in the body of specified reader" << op;
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
  CHECK(!orig_stage.is_scheduled())
      << "Create cache_write before doing split/fuse/reorder";
  compute = orig_stage->op.as<ComputeOpNode>();
  CHECK(compute);
  Array<Expr> args;
  Array<IterVar> new_axis;
  std::unordered_map<const Variable*, Expr> vsub;
  for (IterVar iv : compute->axis) {
    args.push_back(iv->var);
    IterVar new_iv(iv->dom, iv->var->name_hint + ".c");
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

}  // namespace tvm
