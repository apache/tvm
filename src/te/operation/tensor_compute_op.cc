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
 * \brief Tensor Compute Op.
 * \file tensor_compute_op.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <unordered_set>

#include "./op_util.h"
#include "./compute_op.h"
#include "../../arith/compute_expr.h"

namespace tvm {
namespace te {
using namespace tir;
// TensorComputeOpNode
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<TensorComputeOpNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const TensorComputeOpNode*>(node.get());
    p->stream << "tensor_compute_op(" << op->name << ", " << op << ")";
  });

TVM_REGISTER_NODE_TYPE(TensorComputeOpNode);

int TensorComputeOpNode::num_outputs() const {
  return static_cast<int>(this->intrin->buffers.size() - this->inputs.size());
}

DataType TensorComputeOpNode::output_dtype(size_t i) const {
  return this->intrin->buffers[this->inputs.size() + i]->dtype;
}

Operation TensorComputeOpNode::make(std::string name,
                                    std::string tag,
                                    Array<IterVar> axis,
                                    Array<IterVar> reduce_axis,
                                    int schedulable_ndim,
                                    TensorIntrin intrin,
                                    Array<Tensor> tensors,
                                    Array<Region> regions,
                                    Array<PrimExpr> scalar_inputs) {
  auto n = make_object<TensorComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->axis = std::move(axis);
  n->reduce_axis = std::move(reduce_axis);
  n->schedulable_ndim = std::move(schedulable_ndim);
  n->intrin = std::move(intrin);
  n->inputs = std::move(tensors);
  n->input_regions = std::move(regions);
  n->scalar_inputs = std::move(scalar_inputs);
  return Operation(n);
}

TVM_REGISTER_GLOBAL("te.TensorComputeOp")
.set_body_typed(TensorComputeOpNode::make);


Array<Tensor> TensorComputeOpNode::InputTensors() const {
  return inputs;
}

Operation TensorComputeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  auto n = make_object<TensorComputeOpNode>(*this);
  auto intrin = make_object<TensorIntrinNode>(*(this->intrin.operator->()));
  intrin->body = ReplaceTensor(this->intrin->body, rmap);
  if (intrin->reduce_init.defined()) {
    intrin->reduce_init = ReplaceTensor(this->intrin->reduce_init, rmap);
  }
  if (intrin->reduce_update.defined()) {
    intrin->reduce_update = ReplaceTensor(this->intrin->reduce_update, rmap);
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
    arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
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

size_t TensorComputeOpNode::num_schedulable_dims() const {
  return schedulable_ndim;
}

Stmt TensorComputeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);

  // Start bind data.
  Stmt nop = EvaluateNode::make(0);
  std::vector<Stmt> input_bind_nest, output_bind_nest;
  Array<Tensor> inputs = this->InputTensors();

  // input binding
  size_t num_inputs = inputs.size();
  for (size_t i = 0; i < num_inputs; ++i) {
    Tensor tensor = inputs[i];
    Region region = this->input_regions[i];
    Buffer buffer = this->intrin->buffers[i];
    Array<ObjectRef> bind_spec{buffer, tensor};

    Array<PrimExpr> tuple;
    for (size_t i = 0; i < region.size(); ++i) {
      tuple.push_back(region[i]->min);
      tuple.push_back(region[i]->extent);
    }
    input_bind_nest.emplace_back(AttrStmtNode::make(
        bind_spec, tir::attr::buffer_bind_scope,
        CallNode::make(DataType::Handle(),
                       tir::intrinsic::tvm_tuple,
                       tuple, CallNode::Intrinsic), nop));
  }

  // output binding
  for (int i = 0; i < this->num_outputs(); ++i) {
    Tensor tensor = stage->op.output(i);
    Buffer buffer = this->intrin->buffers[num_inputs + i];
    Array<ObjectRef> bind_spec{buffer, tensor};

    Array<PrimExpr> tuple;
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

    output_bind_nest.emplace_back(AttrStmtNode::make(
        bind_spec, tir::attr::buffer_bind_scope,
        CallNode::make(DataType::Handle(),
                       tir::intrinsic::tvm_tuple,
                       tuple, CallNode::Intrinsic), nop));
  }

  // Check variable remap
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  tir::ArgBinder binder(&vmap);

  // Map the expressions passed in the call to the TensorIntrin, to the placeholder
  // variables
  Array<PrimExpr> user_expr = this->scalar_inputs;
  Array<Var> scalar_params = this->intrin->scalar_params;
  Array<PrimExpr> sp_expr;
  for (auto sp : scalar_params) {
    PrimExpr esp = sp;
    sp_expr.push_back(esp);
  }
  CHECK_EQ(sp_expr.size(), user_expr.size());
  // TODO(jdavies-huawei): what name should be used here?
  binder.BindArray(sp_expr, user_expr, this->name);

  size_t tloc = stage->leaf_iter_vars.size();
  ComputeLoopNest n = ComputeLoopNest::make(this, stage, dom_map, debug_keep_trivial_loop);

  if (this->reduce_axis.size() == 0) {
    std::vector<std::vector<Stmt> > nest(
        n.main_nest.begin(), n.main_nest.begin() + tloc + 1);
    nest.emplace_back(MakeIfNest(n.main_predicates));
    CHECK_EQ(n.init_predicates.size(), 0U);
    CHECK(this->intrin->body.defined())
        << "Normal store op for intrin " << this << " is not defined";
    Stmt body = MergeNest(output_bind_nest, this->intrin->body);
    body = MergeNest(input_bind_nest, body);
    body = tir::Substitute(body, vmap);
    body = MergeNest(binder.asserts(), body);
    body = te::Substitute(body, n.main_vmap);
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
    update_nest.emplace_back(MakeIfNest(n.main_predicates));

    if (this->intrin->reduce_init.defined()) {
      // init nest
      std::vector<std::vector<Stmt> > init_nest(
          n.init_nest.begin(), n.init_nest.begin() + tloc + 1);
      init_nest.emplace_back(MakeIfNest(n.init_predicates));
      Stmt init = MergeNest(output_bind_nest, this->intrin->reduce_init);
      init = te::Substitute(init, n.init_vmap);
      init = MergeNest(init_nest, init);
      // The update
      Stmt update = MergeNest(output_bind_nest, this->intrin->reduce_update);
      update = MergeNest(input_bind_nest, update);
      update = tir::Substitute(update, vmap);
      update = MergeNest(binder.asserts(), update);
      update = te::Substitute(update, n.main_vmap);
      update = MergeNest(update_nest, update);
      return MergeNest(common, SeqStmt::Flatten(init, update));
    } else {
      // When init op is not available, use body op for reset in the first iter.
      CHECK(this->intrin->body.defined())
          << "Normal body op is not defined";
      Stmt update = TransformUpdate(stage, dom_map, n,
                                    this->intrin->body,
                                    this->intrin->reduce_update);
      update = MergeNest(output_bind_nest, update);
      update = MergeNest(input_bind_nest, update);
      update = tir::Substitute(update, vmap);
      update = MergeNest(binder.asserts(), update);
      update = te::Substitute(update, n.main_vmap);
      update = MergeNest(update_nest, update);
      return MergeNest(common, update);
    }
  }
}
}  // namespace te
}  // namespace tvm
