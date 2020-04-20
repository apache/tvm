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
 * \brief Scan Operator.
 * \file scan_op.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include "op_util.h"
#include "../schedule/graph.h"

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ScanOpNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const ScanOpNode*>(node.get());
    p->stream << "scan(" << op->name << ", " << op << ")";
});
TVM_REGISTER_NODE_TYPE(ScanOpNode);

int ScanOpNode::num_outputs() const {
  return static_cast<int>(update.size());
}
Array<IterVar> ScanOpNode::root_iter_vars() const {
  Array<IterVar> ret{scan_axis};
  for (IterVar iv : spatial_axis_) {
    ret.push_back(iv);
  }
  return ret;
}

DataType ScanOpNode::output_dtype(size_t i) const {
  return update[i]->dtype;
}

Array<PrimExpr> ScanOpNode::output_shape(size_t i) const {
  CHECK_LT(i, state_placeholder.size());
  return state_placeholder[i]->shape;
}

Operation ScanOpNode::make(std::string name,
                           std::string tag,
                           Map<std::string, ObjectRef> attrs,
                           IterVar axis,
                           Array<Tensor> init,
                           Array<Tensor> update,
                           Array<Tensor> state_placeholder,
                           Array<Tensor> inputs) {
  if (!attrs.defined()) {
    attrs = Map<std::string, ObjectRef>();
  }
  auto n = make_object<ScanOpNode>();
  CHECK_EQ(init.size(), update.size());
  CHECK_EQ(init.size(), state_placeholder.size());
  arith::Analyzer analyzer;
  auto prove_equal = [&](PrimExpr lhs, PrimExpr rhs) {
    return is_zero(analyzer.Simplify(lhs - rhs));
  };

  for (size_t i = 0; i < init.size(); ++i) {
    CHECK_EQ(init[i]->dtype, state_placeholder[i]->dtype);
    CHECK_EQ(init[i]->dtype, update[i]->dtype);
    CHECK(prove_equal(init[i]->shape[0], axis->dom->min))
        << "init.shape[0] need to match scan_axis.dom.min";
    CHECK(prove_equal(
        state_placeholder[i]->shape[0], axis->dom->min + axis->dom->extent))
        << "state_placeholder.shape[0] need to match"
        << " scan_axis.dom.min + scan_axis.dom.extent";
    CHECK_EQ(state_placeholder[i].ndim(), init[i].ndim())
        << "The dimension of init need to match state_placeholder";
    CHECK_EQ(update[i].ndim(), state_placeholder[i].ndim())
        << "The update.ndim need to be state_placeholder.ndim - 1";
    for (size_t k = 0;  k < update[i].ndim(); ++k) {
      CHECK(prove_equal(
          update[i]->shape[k], state_placeholder[i]->shape[k]));
      if (k != 0) {
        // setup spatial axis
        std::ostringstream spatial_name;
        spatial_name << name << ".out" << i << ".i" << k;
        n->spatial_axis_.push_back(
            IterVarNode::make(
                Range::make_by_min_extent(0, update[i]->shape[k]),
                Var(spatial_name.str()), kOpaque));
      }
    }

    for (size_t k = 1;  k < init[i].ndim(); ++k) {
      CHECK(prove_equal(
          init[i]->shape[k], state_placeholder[i]->shape[k]));
    }
  }
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->scan_axis = std::move(axis);
  n->init = std::move(init);
  n->update = std::move(update);
  n->state_placeholder = std::move(state_placeholder);
  n->inputs = std::move(inputs);
  return Operation(n);
}

TVM_REGISTER_GLOBAL("te.ScanOp")
.set_body_typed(ScanOpNode::make);


Array<Tensor> scan(Array<Tensor> init,
                   Array<Tensor> update,
                   Array<Tensor> state_placeholder,
                   Array<Tensor> inputs,
                   std::string name,
                   std::string tag,
                   Map<std::string, ObjectRef> attrs) {
  IterVar scan_axis =
      IterVarNode::make(
          Range::make_by_min_extent(
              init[0]->shape[0], update[0]->shape[0] - init[0]->shape[0]),
          Var(name + ".idx"), kOrdered);
  Operation op = ScanOpNode::make(
      name, tag, attrs, scan_axis,
      init, update, state_placeholder, inputs);
  Array<Tensor> res;
  for (int i = 0; i < op->num_outputs(); ++i) {
    res.push_back(op.output(i));
  }
  return res;
}

Array<Tensor> ScanOpNode::InputTensors() const {
  Array<Tensor> ret;
  for (Tensor t : init) {
    ret.push_back(t);
  }
  for (Tensor t : update) {
    ret.push_back(t);
  }
  return ret;
}

Operation ScanOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  auto n = make_object<ScanOpNode>(*this);
  for (size_t i = 0; i < n->init.size(); ++i) {
    if (rmap.count(n->init[i])) {
      n->init.Set(i, rmap.at(n->init[i]));
    }
    if (rmap.count(n->update[i])) {
      n->update.Set(i, rmap.at(n->update[i]));
    }
  }
  if (!n->init.same_as(init) ||
      !n->update.same_as(update)) {
    return Operation(n);
  } else {
    return self;
  }
}

void ScanOpNode::PropBoundToInputs(
    const Operation& self,
    arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  for (size_t i = 0, sp_idx = 0; i < this->init.size(); ++i) {
    TensorDom* init_dom = nullptr;
    TensorDom* update_dom = nullptr;
    if (out_dom_map->count(this->init[i])) {
      init_dom = &out_dom_map->at(this->init[i]);
    }
    if (out_dom_map->count(this->update[i])) {
      update_dom = &out_dom_map->at(this->update[i]);
    }
    // first dimension, always needed.
    if (init_dom) {
      init_dom->data[0].push_back(IntSet::range(
          Range::make_by_min_extent(0, this->init[i]->shape[0])));
    }
    if (update_dom) {
      update_dom->data[0].push_back(dom_map.at(this->scan_axis->var.get()));
    }
    // The update dimensions
    for (size_t k = 1; k < this->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this->spatial_axis_[sp_idx];
      if (init_dom) {
        init_dom->data[k].push_back(dom_map.at(sp_ax->var.get()));
      }
      if (update_dom) {
        update_dom->data[k].push_back(dom_map.at(sp_ax->var.get()));
      }
    }
  }
}

void ScanOpNode::GatherBound(
    const Operation& self,
    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  CHECK(!out_dom_map->count(this->scan_axis));
  std::vector<Tensor> output(this->num_outputs());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = self.output(i);
  }
  // Update for time axis.
  std::vector<IntSet> time_dom;
  for (size_t i = 0; i < output.size(); ++i) {
    const TensorDom& d = tensor_dom.at(output[i]);
    time_dom.insert(time_dom.end(), d.data[0].begin(), d.data[0].end());
  }
  CHECK(!out_dom_map->count(this->scan_axis));
  arith::Analyzer analyzer;
  Range sdom = this->scan_axis->dom;
  Range r = arith::Union(time_dom).cover_range(sdom);
  (*out_dom_map)[this->scan_axis] = Range::make_by_min_extent(
      sdom->min, analyzer.Simplify(r->extent + r->min - sdom->min));
  Map<IterVar, PrimExpr> fix_pt = ScanFixPointAnalysis(self);
  // Update for spatial axis.
  size_t sp_idx = 0;
  for (size_t i = 0; i < output.size(); ++i) {
    const TensorDom& d = tensor_dom.at(output[i]);
    for (size_t k = 1; k < this->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this->spatial_axis_[sp_idx];
      CHECK(!out_dom_map->count(sp_ax));
      CHECK(fix_pt.count(sp_ax));
      if (fix_pt[sp_ax].as<tir::IntImmNode>()->value) {
        // fix point, we can slice it.
        (*out_dom_map)[sp_ax] = arith::Union(d.data[k]).cover_range(sp_ax->dom);
      } else {
        // not a fix point, need to include everything.
        (*out_dom_map)[sp_ax] = sp_ax->dom;
      }
    }
  }
}

Stmt ScanOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    const Stmt& body) const {
  arith::Analyzer analyzer;
  CHECK_EQ(stage->op.get(), this);
  Range sdom = dom_map.at(this->scan_axis);
  Range tdom = Range::make_by_min_extent(
      0, analyzer.Simplify(sdom->extent + sdom->min));
  Stmt ret = body;
  size_t sp_idx = 0;
  for (size_t i = 0; i < update.size(); ++i) {
    Tensor t = stage->op.output(i);
    CHECK_EQ(static_cast<size_t>(t->value_index), i);
    Region bounds;
    bounds.push_back(tdom);
    for (size_t k = 1; k < this->update[i]->shape.size(); ++k, ++sp_idx) {
      IterVar sp_ax = this->spatial_axis_[sp_idx];
      bounds.push_back(dom_map.at(sp_ax));
    }
    ret = tir::RealizeNode::make(t->op, t->value_index, t->dtype,
                            bounds, const_true(), ret);
  }
  return ret;
}

Stmt ScanOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt provide = AttrStmtNode::make(
      stage->op, tir::attr::scan_update_scope, this->scan_axis->var,
      EvaluateNode::make(0));
  Stmt init = AttrStmtNode::make(
      stage->op, tir::attr::scan_init_scope, 0,
      EvaluateNode::make(0));
  size_t begin_scan = 0;
  for (size_t  i = 0; i < stage->leaf_iter_vars.size(); ++i) {
    if (stage->leaf_iter_vars[i]->iter_type == kThreadIndex) {
      CHECK_EQ(begin_scan, i);
      begin_scan = i + 1;
    }
  }
  std::unordered_map<IterVar, PrimExpr> vmap;
  std::unordered_set<IterVar> empty;
  auto nest = MakeLoopNest(
      stage, dom_map, 0, false, empty, &vmap, debug_keep_trivial_loop);
  nest[begin_scan].push_back(init);
  nest.push_back(
      MakeIfNest(
          MakeBoundCheck(stage, dom_map, vmap, false, empty)));
  return MergeNest(nest, provide);
}
}  // namespace te
}  // namespace tvm
