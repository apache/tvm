/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Compute Op.
 * \file compute_op.cc
 */
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include "./op_util.h"
#include "../schedule/message_passing.h"

namespace tvm {

using namespace ir;

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ComputeOpNode>([](const ComputeOpNode *op, IRPrinter *p) {
    p->stream << "compute(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

int ComputeOpNode::num_outputs() const {
  return body.size();
}

Array<IterVar> ComputeOpNode::root_iter_vars() const {
  if (reduce_axis.size() == 0) return axis;
  Array<IterVar> ret = axis;
  for (IterVar iv : reduce_axis) {
    ret.push_back(iv);
  }
  return ret;
}

Type ComputeOpNode::output_dtype(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  return body[idx].type();
}

Array<Expr> ComputeOpNode::output_shape(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  // for now, all outputs of ComputeOp have the same shape
  std::vector<Expr> shape;
  for (size_t i = 0; i < axis.size(); ++i) {
    const Range& r = axis[i]->dom;
    shape.push_back(r->extent);
  }
  return Array<Expr>(shape);
}

Tensor compute(Array<Expr> shape, FCompute fcompute, std::string name) {
  auto op_node = std::make_shared<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVarNode::make(
        Range(0, shape[i]), Var(os.str(), shape[i].type()), kDataPar));
    args.push_back(axis.back()->var);
  }

  return ComputeOpNode::make(name, axis, {fcompute(args)}).output(0);
}

Operation ComputeOpNode::make(std::string name,
                              Array<IterVar> axis,
                              Array<Expr> body) {
  auto n = std::make_shared<ComputeOpNode>();
  n->name = name;
  n->axis = axis;
  n->body = body;
  if (n->body[0]->is_type<ir::Reduce>()) {
    // batch reduction should have the same axis
    n->reduce_axis = n->body[0].as<ir::Reduce>()->axis;
  }
  return Operation(n);
}

// The schedule related logics
Array<Tensor> ComputeOpNode::InputTensors() const {
  Array<Tensor> ret;
  std::unordered_set<Tensor> visited;
  for (auto& e : body) {
    ir::PostOrderVisit(e, [&ret, &visited](const NodeRef& n) {
        const ir::Call *call = n.as<ir::Call>();
        if (call != nullptr && call->func.defined()) {
          Tensor t = Operation(call->func.node_).output(call->value_index);
          if (!visited.count(t)) {
            ret.push_back(t);
            visited.insert(t);
          }
        }
      });
  }
  return ret;
}

Array<Expr> ReplaceTensor(Array<Expr> exprs,
                          const std::unordered_map<Tensor, Tensor>& replace) {
  Array<Expr> ret;
  for (auto& e : exprs) {
    ret.push_back(op::ReplaceTensor(e, replace));
  }
  return ret;
}

Operation ComputeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  Array<Expr> new_body = ReplaceTensor(this->body, rmap);
  if (!IsSame(new_body, this->body)) {
    return ComputeOpNode::make(name, axis, new_body);
  } else {
    return self;
  }
}

void ComputeOpNode::PropBoundToInputs(
    const Operation& self,
    const std::unordered_map<const Variable*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  auto fvisit = [&dom_map, out_dom_map](const NodeRef& n) {
    auto *call = n.as<ir::Call>();
    if (call != nullptr && call->func.defined()) {
      Tensor t = Operation(call->func.node_).output(call->value_index);
      if (t->op.defined() && out_dom_map->count(t)) {
        TensorDom& dom = out_dom_map->at(t);
        for (size_t i = 0; i < t.ndim(); ++i) {
          dom.data[i].push_back(EvalSet(call->args[i], dom_map));
        }
      }
    }
  };
  for (auto& e : body) ir::PostOrderVisit(e, fvisit);
}

void ComputeOpNode::GatherBound(
    const Operation& self,
    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  const TensorDom& tdom = tensor_dom.at(self.output(0));
  for (size_t i = 0; i < this->axis.size(); ++i) {
    Range r = arith::Union(tdom.data.at(i)).cover_range(this->axis[i]->dom);
    CHECK(!out_dom_map->count(this->axis[i]));
    (*out_dom_map)[this->axis[i]] = r;
  }
  for (size_t i = 0; i < this->reduce_axis.size(); ++i) {
    CHECK(!out_dom_map->count(this->reduce_axis[i]));
    (*out_dom_map)[this->reduce_axis[i]] = this->reduce_axis[i]->dom;
  }
}

Stmt ComputeOpNode::BuildRealize(
    const Operation& self,
    const std::unordered_map<IterVar, Range>& realize_map,
    const Stmt& realize_body) const {
  CHECK_EQ(self.operator->(), this);
  Halide::Internal::Region bounds;
  for (IterVar iv : this->axis) {
    bounds.push_back(realize_map.at(iv));
  }
  Stmt realize = realize_body;
  for (int i = self->num_outputs(); i > 0; --i) {
    Tensor t = self.output(i-1);
    realize = ir::Realize::make(t->op, t->value_index,
      t->dtype, bounds, const_true(), realize);
  }
  return realize;
}

// Build a reduction body.
void MakeReduction(const ComputeOpNode* op,
                   const Tensor& t,
                   Stmt* init,
                   Stmt* provide) {
  Array<Expr>  args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  const Reduce* reduce = op->body[t->value_index].as<Reduce>();
  CHECK(reduce);
  const CommReducerNode* combiner = reduce->combiner.as<CommReducerNode>();
  CHECK(combiner);
  Expr init_value = combiner->identity_element;
  Expr update_value = (*combiner)(t(args), reduce->source);
  *init = Provide::make(t->op, t->value_index, init_value, args);
  *provide = Provide::make(t->op, t->value_index, update_value, args);
  if (!is_one(reduce->condition)) {
    *provide = IfThenElse::make(reduce->condition, *provide);
  }
}

Stmt Substitute(Stmt s,
                const std::unordered_map<IterVar, Expr>& value_map) {
  Map<Var, Expr> temp;
  for (const auto& kv : value_map) {
    temp.Set(kv.first->var, kv.second);
  }
  return ir::Substitute(s, temp);
}

std::vector<Stmt> Substitute(std::vector<Stmt> stmt,
                             const std::unordered_map<IterVar, Expr>& value_map) {
  Map<Var, Expr> temp;
  for (const auto& kv : value_map) {
    temp.Set(kv.first->var, kv.second);
  }
  std::vector<Stmt> ret;
  for (auto& s : stmt)  {
    ret.push_back(ir::Substitute(s, temp));
  }
  return ret;
}

// Cross Thread reduction marker.
bool IsCrossThreadReduction(const ComputeOpNode* self,
                            const Stage& stage) {
  // Verify correctness of leaf nest.
  int normal_red = 0, thread_red = 0;
  for (IterVar iv : stage->leaf_iter_vars) {
    if (iv->iter_type == kCommReduce) {
      auto it = stage->iter_var_attrs.find(iv);
      if (it != stage->iter_var_attrs.end() &&
          (*it).second->bind_thread.defined()) {
        ++thread_red;
      } else {
        ++normal_red;
      }
    } else {
      CHECK_EQ(thread_red, 0)
          << "Cross thread reduce cannot swap with normal data axis";
    }
  }
  CHECK(normal_red == 0 || thread_red == 0)
      << "Cannot mix normal reduction with thread reduce";
  return thread_red != 0;
}

Stmt MakeCrossThreadReduction(
    const ComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map) {
  Array<Expr>  args;
  for (IterVar iv : self->axis) {
    args.push_back(iv->var);
  }
  std::unordered_map<IterVar, Expr> value_map;
  auto nest = op::MakeLoopNest(
      stage, dom_map, 0, false, std::unordered_set<IterVar>(), &value_map);
  auto conds = op::MakeBoundCheck(
      stage, dom_map, false,
      std::unordered_set<IterVar>(), value_map);

  std::vector<Stmt> reduction_bodies;
  for (size_t idx = 0; idx < self->body.size(); ++idx) {
    const Reduce* reduce = self->body[idx].as<Reduce>();
    CHECK(reduce);
    Expr cond = reduce->condition;
    for (Expr v : conds) {
      cond = cond && v;
    }
    Var res_handle("reduce_temp"+std::to_string(idx), Handle());
    Array<Expr> freduce_args;
    freduce_args.push_back(reduce->source);
    freduce_args.push_back(cond);

    for (IterVar iv : stage->leaf_iter_vars) {
      if (iv->iter_type == kCommReduce) {
        auto it = stage->iter_var_attrs.find(iv);
        if (it != stage->iter_var_attrs.end() &&
            (*it).second->bind_thread.defined()) {
          IterVar tv = (*it).second->bind_thread;
          freduce_args.push_back(tv->var);
        }
      }
    }
    // Checks for the thread.
    std::vector<Expr> thread_head_check;
    if (stage->store_predicate.defined()) {
      thread_head_check.emplace_back(stage->store_predicate);
    }
    Type t = reduce->type;
    Expr pred = const_true(t.lanes());
    Stmt reduce_body = Store::make(res_handle,
      Call::make(
        reduce->type,
        ir::intrinsic::tvm_thread_allreduce,
        freduce_args, Call::Intrinsic),
       0, pred);
    reduce_body = AttrStmt::make(
        reduce->combiner,
        attr::reduce_scope,
        make_zero(reduce->type),
        reduce_body);
    Stmt assign_body = Provide::make(
        stage->op, 0, Load::make(reduce->type, res_handle, 0, pred), args);

    assign_body = MergeNest(op::MakeIfNest(thread_head_check), assign_body);
    assign_body = MergeNest(op::MakeIfNest(conds), assign_body);
    Stmt body = Allocate::make(
        res_handle, reduce->type, {1}, const_true(),
        Block::make(reduce_body, assign_body));
    body = AttrStmt::make(
        res_handle, attr::storage_scope, StringImm::make("local"), body);
    body = Substitute(body, value_map);
    reduction_bodies.push_back(body);
  }
  return MergeNest(nest, Block::make(reduction_bodies));
}

Stmt MakeProvide(const ComputeOpNode* op,
                 const Tensor& t) {
  Array<Expr> args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  return Provide::make(t->op, t->value_index, op->body[t->value_index], args);
}

Stmt ComputeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map) const {
  CHECK_EQ(stage->op.operator->(), this);

  if (IsCrossThreadReduction(this, stage)) {
    LOG(INFO) << stage;
    // specially handle cross thread reduction.
    return MakeCrossThreadReduction(this, stage, dom_map);
  }

  std::vector<Stmt> inits;
  std::vector<Stmt> provides;
  if (this->reduce_axis.size() == 0) {
    for (int i = 0; i < this->num_outputs(); ++i) {
      provides.push_back(MakeProvide(this, stage->op.output(i)));
    }
  } else {
    for (int i = 0; i < this->num_outputs(); ++i) {
      Stmt init, provide;
      MakeReduction(this, stage->op.output(i), &init, &provide);
      inits.push_back(init);
      provides.push_back(provide);
    }
  }

  // make loop nest
  std::unordered_map<IterVar, Expr> value_map;
  auto nest = op::MakeLoopNest(
      stage, dom_map, 0, false, std::unordered_set<IterVar>(), &value_map);
  auto preds = op::MakeBoundCheck(stage, dom_map, false,
      std::unordered_set<IterVar>(), value_map);
  for (auto& e : preds) e = likely(e);
  nest.push_back(op::MakeIfNest(preds));
  if (stage->store_predicate.defined()) {
    nest.emplace_back(op::MakeIfNest({stage->store_predicate}));
  }
  provides = Substitute(provides, value_map);

  if (!inits.empty()) {
    // try to find the location to insert the initialization.
    // Fuse the initialization and provide loop when possible.
    std::unordered_map<IterVar, int> update_state;
    for (IterVar iv : this->reduce_axis) {
      update_state[iv] = 2;
    }
    for (IterVar iv : this->axis) {
      update_state[iv] = 1;
    }
    // find which iter var is related to reduction and which is related to axis.
    schedule::PassDownBitMaskOr(stage, &update_state);
    auto leaf_iter_vars = stage->leaf_iter_vars;
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
    std::unordered_set<IterVar> skip_iter;
    for (auto kv : update_state) {
      int flag = kv.second;
      if ((flag & 1) == 0) skip_iter.insert(kv.first);
    }
    auto init_nest = op::MakeLoopNest(
        stage, dom_map, begin_loop, true,
        skip_iter, &init_value_map);
    auto preds = op::MakeBoundCheck(stage, dom_map, true, skip_iter, init_value_map);
    for (auto& e : preds) e = likely(e);
    init_nest.push_back(op::MakeIfNest(preds));
    inits = Substitute(inits, init_value_map);
    Stmt init = MergeNest(init_nest, Block::make(inits));
    // common nest
    std::vector<std::vector<Stmt> > common(nest.begin(), nest.begin() + begin_loop + 1);
    std::vector<std::vector<Stmt> > reduce(nest.begin() + begin_loop + 1, nest.end());
    Stmt provide = MergeNest(reduce, Block::make(provides));
    return MergeNest(common, Block::make(init, provide));
  } else {
    return MergeNest(nest, Block::make(provides));
  }
}
}  // namespace tvm
