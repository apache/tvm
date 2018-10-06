/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Tensor Compute Op.
 * \file tensor_compute_op.cc
 */
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include "./op_util.h"
#include "./compute_op.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
using namespace ir;
// TensorComputeOpNode
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<TensorComputeOpNode>([](const TensorComputeOpNode *op,
                                      IRPrinter *p) {
    p->stream << "tensor_compute_op(" << op->name << ", " << op << ")";
  });

TVM_REGISTER_NODE_TYPE(TensorComputeOpNode);

int TensorComputeOpNode::num_outputs() const {
  return static_cast<int>(this->intrin->buffers.size() - this->inputs.size());
}

Array<IterVar> TensorComputeOpNode::root_iter_vars() const {
  Array<IterVar> ret = axis;
  for (IterVar iv : reduce_axis) {
    ret.push_back(iv);
  }
  return ret;
}

Type TensorComputeOpNode::output_dtype(size_t i) const {
  return this->intrin->buffers[this->inputs.size() + i]->dtype;
}

Array<Expr> TensorComputeOpNode::output_shape(size_t i) const {
  Array<Expr> shape;
  for (const auto& ivar : this->axis) {
    shape.push_back(ivar->dom->extent);
  }
  return shape;
}


Operation TensorComputeOpNode::make(std::string name,
                                    std::string tag,
                                    Array<IterVar> axis,
                                    Array<IterVar> reduce_axis,
                                    int schedulable_ndim,
                                    TensorIntrin intrin,
                                    Array<Tensor> tensors,
                                    Array<Region> regions) {
  auto n = make_node<TensorComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->axis = std::move(axis);
  n->reduce_axis = std::move(reduce_axis);
  n->schedulable_ndim = std::move(schedulable_ndim);
  n->intrin = std::move(intrin);
  n->inputs = std::move(tensors);
  n->input_regions = std::move(regions);
  return Operation(n);
}

Array<Tensor> TensorComputeOpNode::InputTensors() const {
  return inputs;
}

Operation TensorComputeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  auto n = make_node<TensorComputeOpNode>(*this);
  auto intrin = make_node<TensorIntrinNode>(*(this->intrin.operator->()));
  intrin->body = op::ReplaceTensor(this->intrin->body, rmap);
  if (intrin->reduce_init.defined()) {
    intrin->reduce_init = op::ReplaceTensor(this->intrin->reduce_init, rmap);
  }
  if (intrin->reduce_update.defined()) {
    intrin->reduce_update = op::ReplaceTensor(this->intrin->reduce_update, rmap);
  }
  for (size_t i = 0; i < n->inputs.size(); ++i) {
    Tensor t = n->inputs[i];
    if (rmap.count(t)) {
      n->inputs.Set(i, rmap.at(t));
    }
  }

  if (intrin->body.same_as(n->intrin->body) &&
      intrin->reduce_init.same_as(n->intrin->reduce_init) &&
      intrin->reduce_update.same_as(n->intrin->reduce_update) &&
      inputs.same_as(n->inputs)) {
    return self;
  } else {
    n->intrin = TensorIntrin(intrin);
    return Operation(n);
  }
}

void TensorComputeOpNode::PropBoundToInputs(
    const Operation& self,
    const std::unordered_map<const Variable*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  for (size_t i = 0; i < this->inputs.size(); ++i) {
    Tensor t = this->inputs[i];
    Region region = input_regions[i];

    auto it = out_dom_map->find(t);
    if (it == out_dom_map->end()) continue;
    TensorDom& dom = it->second;
    for (size_t j = 0; j < t.ndim(); ++j) {
      dom.data[j].emplace_back(EvalSet(region[j], dom_map));
    }
  }
}

void TensorComputeOpNode::GatherBound(
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

Stmt TensorComputeOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& realize_map,
    const Stmt& body) const {
  CHECK_EQ(stage->op.get(), this);
  HalideIR::Internal::Region bounds;
  for (IterVar iv : this->axis) {
    bounds.push_back(realize_map.at(iv));
  }
  Stmt realize = body;
  for (int i = this->num_outputs(); i > 0; --i) {
    Tensor t = stage->op.output(i-1);
    realize = ir::Realize::make(t->op, t->value_index,
      t->dtype, bounds, const_true(), realize);
    // alignment requirement, only useful for compute
    for (int i = 0; i < schedulable_ndim; ++i) {
      auto it = stage->iter_var_attrs.find(this->axis[i]);
      if (it != stage->iter_var_attrs.end()) {
        IterVarAttr attr = (*it).second;
        if (attr->dim_align_factor != 0) {
          Array<Expr> tuple = {static_cast<int>(i),
                               attr->dim_align_factor,
                               attr->dim_align_offset};
          realize = ir::AttrStmt::make(
              t, ir::attr::buffer_dim_align,
              Call::make(Handle(), ir::intrinsic::tvm_tuple, tuple, Call::Intrinsic),
              realize);
        }
      }
    }
  }
  return realize;
}

ComputeLoopNest MakeLoopNest(
    const TensorComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) {
  CHECK_EQ(stage->op.operator->(), self);
  ComputeLoopNest ret;
  // make main loop nest
  ret.main_nest = op::MakeLoopNest(
      stage, dom_map, 0, false, std::unordered_set<IterVar>(), &ret.main_vmap,
      debug_keep_trivial_loop);
  ret.main_predicates = schedule::MakeBoundCheck(
      stage, dom_map, ret.main_vmap, false,
      std::unordered_set<IterVar>());
  for (auto& e : ret.main_predicates) {
    e = likely(e);
  }
  if (stage->store_predicate.defined()) {
    ret.main_predicates.push_back(stage->store_predicate);
  }
  if (self->reduce_axis.size() != 0) {
    // try to find the location to insert the initialization.
    // Fuse the initialization and provide loop when possible.
    std::unordered_map<IterVar, int> update_state;
    for (IterVar iv : self->reduce_axis) {
      update_state[iv] = 2;
    }
    for (int i = 0; i < self->schedulable_ndim; ++i) {
      update_state[self->axis[i]] = 1;
    }
    // find which iter var is related to reduction and which is related to axis.
    schedule::PassDownBitMaskOr(stage, &update_state);
    auto leaf_iter_vars = stage->leaf_iter_vars;
    // first first loop that is related to reduction.
    size_t begin_loop = leaf_iter_vars.size();
    for (size_t i = 0; i < leaf_iter_vars.size(); ++i) {
      auto iv = leaf_iter_vars[i];
      int flag = update_state.at(iv);
      if ((flag & 2) != 0) {
        begin_loop = i; break;
      }
      ret.init_vmap[iv] = ret.main_vmap.at(iv);
    }
    ret.num_common_loop = begin_loop;
    // skip loops that does not relates to axis.
    std::unordered_set<IterVar> skip_iter;
    for (auto kv : update_state) {
      int flag = kv.second;
      if ((flag & 1) == 0) skip_iter.insert(kv.first);
    }
    ret.init_nest = op::MakeLoopNest(
        stage, dom_map, begin_loop, true,
        skip_iter, &(ret.init_vmap), debug_keep_trivial_loop);
    ret.init_predicates = schedule::MakeBoundCheck(
        stage, dom_map, ret.init_vmap, true, skip_iter);
    for (auto& e : ret.init_predicates) {
      e = likely(e);
    }
  } else {
    CHECK_EQ(ret.main_nest.size(), stage->leaf_iter_vars.size() + 1);
    ret.num_common_loop = stage->leaf_iter_vars.size();
  }
  // copy elison here.
  return ret;
}


Stmt TensorComputeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);

  // Start bind data.
  Stmt nop = Evaluate::make(0);
  std::vector<Stmt> input_bind_nest, output_bind_nest;
  Array<Tensor> inputs = this->InputTensors();

  // input binding
  size_t num_inputs = inputs.size();
  for (size_t i = 0; i < num_inputs; ++i) {
    Tensor tensor = inputs[i];
    Region region = this->input_regions[i];
    Buffer buffer = this->intrin->buffers[i];
    Array<NodeRef> bind_spec{buffer, tensor};

    Array<Expr> tuple;
    for (size_t i = 0; i < region.size(); ++i) {
      tuple.push_back(region[i]->min);
      tuple.push_back(region[i]->extent);
    }
    input_bind_nest.emplace_back(AttrStmt::make(
        bind_spec, ir::attr::buffer_bind_scope,
        Call::make(Handle(), ir::intrinsic::tvm_tuple, tuple, Call::Intrinsic), nop));
  }

  // output binding
  for (int i = 0; i < this->num_outputs(); ++i) {
    Tensor tensor = stage->op.output(i);
    Buffer buffer = this->intrin->buffers[num_inputs + i];
    Array<NodeRef> bind_spec{buffer, tensor};

    Array<Expr> tuple;
    for (size_t i = 0; i < this->axis.size(); ++i) {
      auto ivar = this->axis[i];
      if (i < static_cast<size_t>(this->schedulable_ndim)) {
        tuple.push_back(ivar->var);
        tuple.push_back(1);
      } else {
        Range dom = ivar->dom;
        tuple.push_back(dom->min);
        tuple.push_back(dom->extent);
      }
    }

    output_bind_nest.emplace_back(AttrStmt::make(
        bind_spec, ir::attr::buffer_bind_scope,
        Call::make(Handle(), ir::intrinsic::tvm_tuple, tuple, Call::Intrinsic), nop));
  }

  // Check variable remap
  std::unordered_map<const Variable*, Expr> vmap;
  ir::ArgBinder binder(&vmap);

  size_t tloc = stage->leaf_iter_vars.size();
  ComputeLoopNest n = MakeLoopNest(this, stage, dom_map, debug_keep_trivial_loop);

  if (this->reduce_axis.size() == 0) {
    std::vector<std::vector<Stmt> > nest(
        n.main_nest.begin(), n.main_nest.begin() + tloc + 1);
    nest.emplace_back(op::MakeIfNest(n.main_predicates));
    CHECK_EQ(n.init_predicates.size(), 0U);
    CHECK(this->intrin->body.defined())
        << "Normal store op for intrin " << this << " is not defined";
    Stmt body = MergeNest(output_bind_nest, this->intrin->body);
    body = MergeNest(input_bind_nest, body);
    body = ir::Substitute(body, vmap);
    body = MergeNest(binder.asserts(), body);
    body = op::Substitute(body, n.main_vmap);
    Stmt ret =  MergeNest(nest, body);
    return ret;
  } else {
    // Need to split reduction
    CHECK(this->intrin->reduce_update.defined())
        << "Reduction update op is not defined";
    // Need init and update steps
    CHECK_NE(this->reduce_axis.size(), 0U);
    std::vector<std::vector<Stmt> > common(
        n.main_nest.begin(), n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt> > update_nest(
        n.main_nest.begin() + n.num_common_loop + 1, n.main_nest.begin() + tloc + 1);
    update_nest.emplace_back(op::MakeIfNest(n.main_predicates));

    if (this->intrin->reduce_init.defined()) {
      // init nest
      std::vector<std::vector<Stmt> > init_nest(
          n.init_nest.begin(), n.init_nest.begin() + tloc + 1);
      init_nest.emplace_back(op::MakeIfNest(n.init_predicates));
      Stmt init = MergeNest(output_bind_nest, this->intrin->reduce_init);
      init = op::Substitute(init, n.init_vmap);
      init = MergeNest(init_nest, init);
      // The update
      Stmt update = MergeNest(output_bind_nest, this->intrin->reduce_update);
      update = MergeNest(input_bind_nest, update);
      update = ir::Substitute(update, vmap);
      update = MergeNest(binder.asserts(), update);
      update = op::Substitute(update, n.main_vmap);
      update = MergeNest(update_nest, update);
      return MergeNest(common, Block::make(init, update));
    } else {
      // When init op is not available, use body op for reset in the first iter.
      CHECK(this->intrin->body.defined())
          << "Normal body op is not defined";
      Stmt update = TransformUpdate(stage, dom_map, n,
                                    this->intrin->body,
                                    this->intrin->reduce_update);
      update = MergeNest(output_bind_nest, update);
      update = MergeNest(input_bind_nest, update);
      update = ir::Substitute(update, vmap);
      update = MergeNest(binder.asserts(), update);
      update = op::Substitute(update, n.main_vmap);
      update = MergeNest(update_nest, update);
      return MergeNest(common, update);
    }
  }
}

}  // namespace tvm
