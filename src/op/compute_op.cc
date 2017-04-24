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
  return 1;
}

Array<IterVar> ComputeOpNode::root_iter_vars() const {
  if (reduce_axis.size() == 0) return axis;
  Array<IterVar> ret = axis;
  for (IterVar iv : reduce_axis) {
    ret.push_back(iv);
  }
  return ret;
}

Type ComputeOpNode::output_dtype(size_t i) const {
  CHECK_EQ(i, 0U);
  return body.type();
}

Array<Expr> ComputeOpNode::output_shape(size_t i) const {
  CHECK_EQ(i, 0U);
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

  return ComputeOpNode::make(name, axis, fcompute(args)).output(0);
}

Operation ComputeOpNode::make(std::string name,
                              Array<IterVar> axis,
                              Expr body) {
  auto n = std::make_shared<ComputeOpNode>();
  n->name = name;
  n->axis = axis;
  n->body = body;
  if (n->body->is_type<ir::Reduce>()) {
    n->reduce_axis = n->body.as<ir::Reduce>()->axis;
  }
  return Operation(n);
}

// The schedule related logics
Array<Tensor> ComputeOpNode::InputTensors() const {
  Array<Tensor> ret;
  std::unordered_set<Tensor> visited;
  ir::PostOrderVisit(body, [&ret, &visited](const NodeRef& n) {
      const ir::Call *call = n.as<ir::Call>();
      if (call != nullptr && call->func.defined()) {
        Tensor t = Operation(call->func.node_).output(call->value_index);
        if (!visited.count(t)) {
          ret.push_back(t);
          visited.insert(t);
        }
      }
    });
  return ret;
}

Operation ComputeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  Expr new_body = op::ReplaceTensor(this->body, rmap);
  if (!new_body.same_as(this->body)) {
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
  ir::PostOrderVisit(body, fvisit);
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
  Tensor t = self.output(0);
  Halide::Internal::Region bounds;
  for (IterVar iv : this->axis) {
    bounds.push_back(realize_map.at(iv));
  }
  return ir::Realize::make(t->op, t->value_index, t->dtype,
                           bounds, const_true(), realize_body);
}

// Build a reduction body.
void MakeReduction(const ComputeOpNode* op,
                   const Tensor& t,
                   Stmt* init,
                   Stmt* provide) {
  Stmt no_op = Evaluate::make(0);
  std::vector<Stmt> nest;
  Array<Expr>  args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  const Reduce* reduce = op->body.as<Reduce>();
  CHECK(reduce);
  Expr init_value = reduce->identity_element;
  Expr update_value = reduce->combiner(t(args), reduce->source);
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
  const Reduce* reduce = self->body.as<Reduce>();
  CHECK(reduce);
  std::unordered_map<IterVar, Expr> value_map;
  auto nest = op::MakeLoopNest(
      stage, dom_map, 0, false, std::unordered_set<IterVar>(), &value_map);
  auto conds = op::MakeBoundCheck(
      stage, dom_map, false,
      std::unordered_set<IterVar>(), value_map);
  Expr cond = reduce->condition;
  for (Expr v : conds) {
    cond = cond && v;
  }
  Var res_handle("reduce_temp", Handle());
  Array<Expr> freduce_args;
  freduce_args.push_back(reduce->source);
  freduce_args.push_back(reduce->identity_element);
  freduce_args.push_back(cond);

  std::vector<Expr> thread_head_check;
  for (IterVar iv : stage->leaf_iter_vars) {
    if (iv->iter_type == kCommReduce) {
      auto it = stage->iter_var_attrs.find(iv);
      if (it != stage->iter_var_attrs.end() &&
          (*it).second->bind_thread.defined()) {
        IterVar tv = (*it).second->bind_thread;
        freduce_args.push_back(tv->var);
        thread_head_check.push_back(tv->var == 0);
      }
    }
  }
  Stmt reduce_body = Store::make(res_handle,
    Call::make(
      reduce->type,
      ir::intrinsic::tvm_thread_allreduce,
      freduce_args, Call::Intrinsic),
    0);
  reduce_body = AttrStmt::make(
    reduce->combiner,
    attr::reduce_scope,
    make_zero(reduce->type),
    reduce_body);
  Stmt assign_body = Provide::make(
      stage->op, 0, Load::make(reduce->type, res_handle, 0), args);
  assign_body = MergeNest(op::MakeIfNest(thread_head_check), assign_body);
  assign_body = MergeNest(op::MakeIfNest(conds), assign_body);
  Stmt body = Allocate::make(
      res_handle, reduce->type, {1}, const_true(),
      Block::make(reduce_body, assign_body));
  body = AttrStmt::make(
      res_handle, attr::storage_scope, StringImm::make("local"), body);
  body = Substitute(body, value_map);
  return MergeNest(nest, body);
}

Stmt MakeProvide(const ComputeOpNode* op,
                 const Tensor& t) {
  Array<Expr> args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  return Provide::make(t->op, t->value_index, op->body, args);
}

Stmt ComputeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map) const {
  CHECK_EQ(stage->op.operator->(), this);

  if (IsCrossThreadReduction(this, stage)) {
    // specially handle cross thread reduction.
    return MakeCrossThreadReduction(this, stage, dom_map);
  }
  Stmt init, provide;
  if (this->reduce_axis.size() == 0) {
    provide = MakeProvide(this, stage->op.output(0));
  } else {
    MakeReduction(this, stage->op.output(0), &init, &provide);
  }
  // make loop nest
  std::unordered_map<IterVar, Expr> value_map;
  auto nest = op::MakeLoopNest(
      stage, dom_map, 0, false, std::unordered_set<IterVar>(), &value_map);
  nest.push_back(op::MakeIfNest(op::MakeBoundCheck(
      stage, dom_map, false,
      std::unordered_set<IterVar>(), value_map)));
  provide = Substitute(provide, value_map);

  if (init.defined()) {
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
    init_nest.push_back(
        op::MakeIfNest(
            op::MakeBoundCheck(stage, dom_map, true, skip_iter, init_value_map)));
    init = Substitute(init, init_value_map);
    init  = MergeNest(init_nest, init);
    // common nest
    std::vector<std::vector<Stmt> > common(nest.begin(), nest.begin() + begin_loop + 1);
    std::vector<std::vector<Stmt> > reduce(nest.begin() + begin_loop + 1, nest.end());
    provide = MergeNest(reduce, provide);
    return MergeNest(common, Block::make(init, provide));
  } else {
    return MergeNest(nest, provide);
  }
}
}  // namespace tvm
