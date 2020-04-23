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
 * \brief Logics related to tensorize, used by ComputeOpNode.
 * \file tensorize.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/analysis.h>
#include <tvm/runtime/registry.h>

#include "op_util.h"
#include "compute_op.h"
#include "../schedule/message_passing.h"

namespace tvm {
namespace te {

using namespace tir;

// Detect the region of input and output to be tensrized.
// out_dom: the domain of root iter vars in output op
// in_region: region of each input tensor.
// return The location of the tensorized scope start.
size_t InferTensorizeRegion(
    const ComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    std::unordered_map<IterVar, Range>* out_dom,
    std::unordered_map<Tensor, Array<Range> >* in_region) {
  // Get the bound of the tensorized scope.
  bool found_point = false;
  size_t loc_scope = 0;
  std::unordered_map<IterVar, IntSet> up_state;
  // Loop over the leafs
  for (size_t i = stage->leaf_iter_vars.size(); i != 0; --i) {
    IterVar iv = stage->leaf_iter_vars[i - 1];
    CHECK(iv->iter_type == kDataPar ||
          iv->iter_type == kCommReduce);
    auto vit = dom_map.find(iv);
    CHECK(vit != dom_map.end());
    const Range& vrange = vit->second;
    if (is_one(vrange->extent)) {
      up_state[iv] = IntSet::single_point(vrange->min);
    } else if (found_point) {
      CHECK(is_zero(vrange->min));
      up_state[iv] = IntSet::single_point(iv->var);
    } else {
      up_state[iv] = IntSet::range(vrange);
    }
    auto iit = stage->iter_var_attrs.find(iv);
    if (iit != stage->iter_var_attrs.end()) {
      const IterVarAttr& attr = (*iit).second;
      if (!found_point) {
        CHECK(!attr->bind_thread.defined())
            << "Do not allow thread in tensorize scope";
      }
      if (attr->iter_type == kTensorized) {
        CHECK(!found_point) << "Do not allow two tensorized point";
        found_point = true;
        loc_scope = i - 1;
      }
    }
  }
  CHECK(found_point);
  // Get domain of the tensorized scope.
  te::PassUpDomain(stage, dom_map, &up_state);
  // Get domains if inputs
  std::unordered_map<Tensor, TensorDom> in_dom;
  std::unordered_map<const VarNode*, IntSet> temp_dmap;
  arith::Analyzer analyzer;
  Array<Tensor> inputs = self->InputTensors();
  for (Tensor t : inputs) {
    in_dom.emplace(t, TensorDom(t.ndim()));
  }
  for (IterVar iv : self->root_iter_vars()) {
    IntSet iset = up_state.at(iv);
    Range iv_range = iset.cover_range(dom_map.at(iv));
    (*out_dom)[iv] = iv_range;
    analyzer.Bind(iv->var, iv_range);
    temp_dmap[iv->var.get()] = iset;
  }
  // Input domains
  self->PropBoundToInputs(stage->op, &analyzer, temp_dmap, &in_dom);
  Range none;
  for (const auto& kv : in_dom) {
    Array<Range> vec;
    const Tensor& t = kv.first;
    for (size_t i = 0; i < t.ndim(); ++i) {
      Range r = arith::Union(kv.second.data.at(i)).cover_range(none);
      CHECK(r.defined()) << "cannot deduce region of tensorized scope for input " << t;
      vec.push_back(std::move(r));
    }
    (*in_region)[t] = std::move(vec);
  }
  return loc_scope;
}

void VerifyTensorizeLoopNest(const ComputeOpNode* self,
                             const Stage& stage,
                             const ComputeLoopNest& n,
                             size_t tloc) {
  // Veirfication step.
  std::unordered_set<const VarNode*> banned;
  CHECK_EQ(n.main_nest.size(), stage->leaf_iter_vars.size() + 1);
  CHECK(n.init_nest.size() == stage->leaf_iter_vars.size() + 1 ||
        n.init_nest.size() == 0);
  auto f_push_banned = [&banned](const Stmt& s) {
    if (const ForNode* op = s.as<ForNode>()) {
        banned.insert(op->loop_var.get());
    } else if (const AttrStmtNode* op = s.as<AttrStmtNode>()) {
      if (const IterVarNode* iv = op->node.as<IterVarNode>()) {
        banned.insert(iv->var.get());
      }
    } else if (const LetStmtNode* op = s.as<LetStmtNode>()) {
      banned.insert(op->var.get());
    }
  };
  for (size_t i = tloc; i < stage->leaf_iter_vars.size(); ++i) {
    for (const Stmt& s : n.main_nest[i + 1]) {
      f_push_banned(s);
    }
    if (n.init_nest.size() != 0) {
      for (const Stmt& s : n.init_nest[i + 1]) {
        f_push_banned(s);
      }
    }
  }

  auto fbanned = [&](const VarNode* node) {
    return banned.count(node);
  };

  for (const PrimExpr& pred : n.main_predicates) {
    if (tir::ExprUseVar(pred, fbanned)) {
      LOG(FATAL) << "Tensorize failed, split condition "
                 << pred << " relies on var defined inside tensorize scope";
    }
  }
  for (const PrimExpr& pred : n.init_predicates) {
    if (tir::ExprUseVar(pred, fbanned)) {
      LOG(FATAL) << "Tensorize failed, split condition "
                 << pred << " relies on var defined inside tensorize scope";
    }
  }
}

// Remap the tensor placeholder, index and inline things.
class TensorIntrinMatcher final : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    if (op->call_type == CallNode::Halide) {
      Tensor t = Downcast<Operation>(op->func).output(op->value_index);
      auto it = in_remap_.find(t);
      if (it != in_remap_.end()) {
        const InputEntry& e = it->second;
        CHECK_EQ(op->args.size(), e.region.size());
        Array<PrimExpr> args;
        for (size_t i = e.start; i < e.region.size(); ++i) {
          args.push_back(op->args[i] - e.region[i]->min);
        }
        return CallNode::make(
            op->dtype, e.tensor->op->name, args,
            op->call_type, e.tensor->op, e.tensor->value_index);
      }
    }
    return expr;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  PrimExpr VisitExpr_(const ReduceNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<ReduceNode>();
    Array<IterVar> axis;
    for (size_t i = 0; i < op->axis.size(); ++i) {
      auto it = axis_remap_.find(op->axis[i]);
      if (it != axis_remap_.end()) {
        axis.push_back(it->second);
      }
    }
    return ReduceNode::make(
        op->combiner, op->source, axis, op->condition, op->value_index);
  }

  void Init(const ComputeOpNode* self,
            const Stage& stage,
            const std::unordered_map<IterVar, Range>& dom_map,
            const std::unordered_map<IterVar, Range>& out_dom,
            const std::unordered_map<Tensor, Array<Range> >& in_region,
            const TensorIntrin& intrin,
            Map<Var, Range>* compute_intrin_iter_space) {
    CHECK(self == stage->op.get());

    for (size_t i = 0; i < stage->leaf_iter_vars.size(); ++i) {
      IterVar iv = stage->leaf_iter_vars[i];
      auto vit = dom_map.find(iv);
      if (vit != dom_map.end()) {
        const Range vrange = vit->second;
        compute_intrin_iter_space->Set(iv->var, vrange);
      }
    }
    analyzer_.Bind(*compute_intrin_iter_space);

    // input remap.
    Array<Tensor> inputs = self->InputTensors();
    CHECK_EQ(inputs.size(), intrin->inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      InputEntry e;
      e.tensor = intrin->inputs[i];
      e.region = Array<Range>(in_region.at(inputs[i]));
      CHECK_GE(e.region.size(), e.tensor.ndim());
      // Enable fuzzy matching, to match [1, n, m] to [n, m]
      e.start = e.region.size() - e.tensor.ndim();
      for (size_t j = 0; j < e.start; ++j) {
        auto canonical_extent = analyzer_.Simplify(e.region[j]->extent);
        CHECK(is_one(canonical_extent))
            << "Tensorize " << intrin->name << ":"
            << " Input dimension mismatch with tensor intrin "
            << " expected shape=" << e.tensor->shape
            << ", given region=" << e.region;
      }
      in_remap_[inputs[i]] = e;
    }
    // output remap
    const ComputeOpNode* intrin_compute = intrin->op.as<ComputeOpNode>();
    CHECK(intrin_compute) << "Only support compute intrinsic for now";
    CHECK_GE(self->axis.size(), intrin_compute->axis.size())
        << "Tensorize: Output mismatch with tensor intrin ";
    // Enable fuzzy matching, to match [1, n, m] to [n, m]
    size_t axis_start = self->axis.size() - intrin_compute->axis.size();
    for (size_t i = 0; i < axis_start; ++i) {
      Range r = out_dom.at(self->axis[i]);
      CHECK(is_one(r->extent))
          << "Tensorize: Output mismatch with tensor intrin "
          << " intrin-dim=" << intrin_compute->axis.size()
          << ", tensorize-dim=" << self->axis.size();
      var_remap_[self->axis[i]->var.get()] = r->min;
    }
    // Assume we tensorize at regin axis i [min, min + extent)
    // The corresponding intrinsic axis is j [0, extent)
    // Remap index i to j + min
    for (size_t i = axis_start; i < self->axis.size(); ++i) {
      IterVar iv = self->axis[i];
      IterVar target_iv = intrin_compute->axis[i - axis_start];
      Range r = out_dom.at(iv);
      var_remap_[iv->var.get()] = target_iv->var + r->min;
      axis_remap_[iv] = target_iv;
      compute_intrin_iter_space->Set(target_iv->var, target_iv->dom);
    }
    // Remap reduction axis
    CHECK_GE(self->reduce_axis.size(), intrin_compute->reduce_axis.size())
        << "Tensorize: Reduction dimension mismatch with tensor intrin";
    axis_start = self->reduce_axis.size() - intrin_compute->reduce_axis.size();
    for (size_t i = 0; i < axis_start; ++i) {
      Range r = out_dom.at(self->reduce_axis[i]);
      CHECK(is_one(r->extent))
          << "Tensorize: Reduction mismatch with tensor intrin "
          << " intrin-dim=" << intrin_compute->reduce_axis.size()
          << ", tensorize-dim=" << self->reduce_axis.size();
      var_remap_[self->reduce_axis[i]->var.get()] = r->min;
    }
    for (size_t i = axis_start; i < self->reduce_axis.size(); ++i) {
      IterVar iv = self->reduce_axis[i];
      IterVar target_iv = intrin_compute->reduce_axis[i - axis_start];
      Range r = out_dom.at(iv);
      var_remap_[iv->var.get()] = target_iv->var + r->min;
      axis_remap_[iv] = target_iv;
      compute_intrin_iter_space->Set(target_iv->var, target_iv->dom);
    }
  }

 private:
  // Input entry
  struct InputEntry {
    Tensor tensor;
    size_t start;
    Array<Range> region;
  };
  // input data remap
  std::unordered_map<Tensor, InputEntry> in_remap_;
  // variable remap.
  std::unordered_map<const VarNode*, PrimExpr> var_remap_;
  // IterVar remap.
  std::unordered_map<IterVar, IterVar> axis_remap_;
  // arith analyzer
  arith::Analyzer analyzer_;
};

// Try to match tensor dataflow of the stage with the intrinsic
Array<PrimExpr> MatchTensorizeBody(
    const ComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    const std::unordered_map<IterVar, Range>& out_dom,
    const std::unordered_map<Tensor, Array<Range> >& in_region,
    const TensorIntrin& intrin,
    Map<Var, Range>* compute_intrin_iter_space) {
  TensorIntrinMatcher matcher;
  matcher.Init(self, stage, dom_map, out_dom, in_region, intrin, compute_intrin_iter_space);
  Array<PrimExpr> ret;
  for (PrimExpr expr : self->body) {
    ret.push_back(matcher(expr));
  }
  return ret;
}

void VerifyTensorizeBody(
    const ComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    const std::unordered_map<IterVar, Range>& out_dom,
    const std::unordered_map<Tensor, Array<Range> >& in_region,
    const TensorIntrin& intrin) {
  StructuralEqual expr_equal;
  Map<Var, Range> compute_intrin_iter_space;
  Array<PrimExpr> body = MatchTensorizeBody(self, stage, dom_map, out_dom, in_region, intrin,
                                        &compute_intrin_iter_space);
  const ComputeOpNode* intrin_compute = intrin->op.as<ComputeOpNode>();
  CHECK(intrin_compute) << "Only support compute intrinsic for now";
  CHECK_EQ(body.size(), intrin_compute->body.size())
      << "Tensorize failed: body size mismatch";
  arith::Analyzer ana;
  ana.Bind(compute_intrin_iter_space);

  for (size_t i = 0; i < body.size(); ++i) {
    PrimExpr lhs = ana.Simplify(body[i]);
    PrimExpr rhs = ana.Simplify(intrin_compute->body[i]);
    if (lhs.dtype() != rhs.dtype()) {
      LOG(FATAL)
          << "Failed to match the data type with TensorIntrin "
          << intrin->name << "'s declaration "
          << " provided=" << lhs.dtype()
          << ", intrin=" << rhs.dtype();
    }
    CHECK(expr_equal(lhs, rhs))
        << "Failed to match the compute with TensorIntrin "
        << intrin->name << "'s declaration "
        << " provided= " << lhs
        << ", intrin=  " << rhs;
  }
}

Stmt MakeTensorize(const ComputeOpNode* self,
                   const Stage& stage,
                   const std::unordered_map<IterVar, Range>& dom_map,
                   bool debug_keep_trivial_loop) {
  std::unordered_map<IterVar, Range> out_dom;
  std::unordered_map<Tensor, Array<Range> > in_region;
  size_t tloc = InferTensorizeRegion(self, stage, dom_map, &out_dom, &in_region);
  TensorIntrin intrin = stage->iter_var_attrs.at(
      stage->leaf_iter_vars[tloc])->tensor_intrin;
  CHECK(intrin.defined());
  ComputeLoopNest n = ComputeLoopNest::make(self, stage, dom_map, debug_keep_trivial_loop);
  VerifyTensorizeLoopNest(self, stage, n, tloc);
  VerifyTensorizeBody(self, stage, dom_map, out_dom, in_region, intrin);
  // Start bind data.
  Stmt nop = EvaluateNode::make(0);
  std::vector<Stmt> input_bind_nest, output_bind_nest;
  Array<Tensor> inputs = self->InputTensors();
  CHECK_EQ(inputs.size(), intrin->inputs.size())
      << "Tensorize failed: input size mismatch ";
  // input binding
  for (size_t i = 0; i < intrin->inputs.size(); ++i) {
    Tensor tensor = inputs[i];
    Buffer buffer = intrin->buffers[i];
    Array<ObjectRef> bind_spec{buffer, tensor};
    auto it = in_region.find(tensor);
    CHECK(it != in_region.end());
    const Array<Range>& region = it->second;
    Array<PrimExpr> tuple;
    for (const Range r : region) {
      tuple.push_back(r->min);
      tuple.push_back(r->extent);
    }
    input_bind_nest.emplace_back(AttrStmtNode::make(
        bind_spec, tir::attr::buffer_bind_scope,
        CallNode::make(DataType::Handle(),
                       tir::intrinsic::tvm_tuple,
                       tuple, CallNode::Intrinsic), nop));
  }
  // output binding
  const ComputeOpNode* intrin_compute = intrin->op.as<ComputeOpNode>();
  CHECK(intrin_compute) << "Only support compute intrinsic for now";
  CHECK_EQ(intrin->inputs.size() + intrin_compute->body.size(), intrin->buffers.size());
  CHECK_EQ(intrin_compute->body.size(), self->body.size());
  Array<PrimExpr> tuple;
  for (IterVar iv : self->axis) {
    auto it = out_dom.find(iv);
    CHECK(it != out_dom.end());
    tuple.push_back(it->second->min);
    tuple.push_back(it->second->extent);
  }
  for (size_t i = intrin->inputs.size(); i < intrin->buffers.size(); ++i) {
    Tensor tensor = stage->op.output(i - intrin->inputs.size());
    Buffer buffer = intrin->buffers[i];
    Array<ObjectRef> bind_spec{buffer, tensor};
    output_bind_nest.emplace_back(AttrStmtNode::make(
        bind_spec, tir::attr::buffer_bind_scope,
        CallNode::make(DataType::Handle(),
                       tir::intrinsic::tvm_tuple,
                       tuple, CallNode::Intrinsic), nop));
  }
  // Check variable remap
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  tir::ArgBinder binder(&vmap);
  CHECK_GE(self->reduce_axis.size(), intrin_compute->reduce_axis.size())
      << "Tensorization fail: reduction axis size do not match";
  size_t start = self->reduce_axis.size() - intrin_compute->reduce_axis.size();
  for (size_t i = 0; i < start; ++i) {
    IterVar iv = self->reduce_axis[i];
    auto it = out_dom.find(iv);
    CHECK(it != out_dom.end());
    CHECK(is_one(it->second->extent))
        << "Tensorization fail: reduction axis size do not match";
  }
  for (size_t i = start; i < self->reduce_axis.size(); ++i) {
    IterVar iv = self->reduce_axis[i];
    IterVar target = intrin_compute->reduce_axis[i - start];
    auto it = out_dom.find(iv);
    CHECK(it != out_dom.end());
    binder.Bind(target->dom->min, make_const(iv->dom->min.dtype(), 0),
                "tensir_intrin.reduction.min");
    binder.Bind(target->dom->extent, it->second->extent,
                "tensir_intrin.reduction.extent");
  }
  if (tloc <= n.num_common_loop) {
    // Do no need to split reduction
    std::vector<std::vector<Stmt> > nest(
        n.main_nest.begin(), n.main_nest.begin() + tloc + 1);
    nest.emplace_back(MakeIfNest(n.main_predicates));
    CHECK_EQ(n.init_predicates.size(), 0U);
    CHECK(intrin->body.defined())
        << "Normal store op for intrin " << intrin << " is not defined";
    Stmt body = MergeNest(output_bind_nest, intrin->body);
    body = MergeNest(input_bind_nest, body);
    body = tir::Substitute(body, vmap);
    body = MergeNest(binder.asserts(), body);
    body = te::Substitute(body, n.main_vmap);
    return MergeNest(nest, body);
  } else {
    // Need to split reduction
    CHECK(intrin->reduce_update.defined())
        << "Reduction update op for intrin " << intrin << " is not defined";
    // Need init and update steps
    CHECK_NE(self->reduce_axis.size(), 0U);
    std::vector<std::vector<Stmt> > common(
        n.main_nest.begin(), n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt> > update_nest(
        n.main_nest.begin() + n.num_common_loop + 1, n.main_nest.begin() + tloc + 1);
    update_nest.emplace_back(MakeIfNest(n.main_predicates));

    if (intrin->reduce_init.defined()) {
      // init nest
      std::vector<std::vector<Stmt> > init_nest(
          n.init_nest.begin(), n.init_nest.begin() + tloc + 1);
      init_nest.emplace_back(MakeIfNest(n.init_predicates));
      Stmt init = MergeNest(output_bind_nest, intrin->reduce_init);
      init = te::Substitute(init, n.init_vmap);
      init = MergeNest(init_nest, init);
      // The update
      Stmt update = MergeNest(output_bind_nest, intrin->reduce_update);
      update = MergeNest(input_bind_nest, update);
      update = tir::Substitute(update, vmap);
      update = MergeNest(binder.asserts(), update);
      update = te::Substitute(update, n.main_vmap);
      update = MergeNest(update_nest, update);
      return MergeNest(common, SeqStmt::Flatten(init, update));
    } else {
      // When init op is not available, use body op for reset in the first iter.
      CHECK(intrin->body.defined())
          << "Normal body op for intrin " << intrin << " is not defined";
      Stmt update = TransformUpdate(stage, dom_map, n,
                                    intrin->body,
                                    intrin->reduce_update);
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

// Register functions for unittests
TVM_REGISTER_GLOBAL("test.op.InferTensorizeRegion")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    Stage stage = args[0];
    Map<IterVar, Range> dmap = args[1];
    std::unordered_map<IterVar, Range> out_dom;
    std::unordered_map<Tensor, Array<Range> > in_region;
    CHECK(stage->op.as<ComputeOpNode>());
    InferTensorizeRegion(stage->op.as<ComputeOpNode>(),
                         stage,
                         as_unordered_map(dmap),
                         &out_dom, &in_region);
    *ret = Array<ObjectRef>{Map<IterVar, Range>(out_dom),
                          Map<Tensor, Array<Range> >(in_region)};
  });

TVM_REGISTER_GLOBAL("test.op.MatchTensorizeBody")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    Stage stage = args[0];
    Map<IterVar, Range> out_dom = args[1];
    Map<Tensor, Array<Range> > in_region = args[2];
    TensorIntrin intrin = args[3];
    Map<Var, Range> vrange;
    CHECK(stage->op.as<ComputeOpNode>());
    *ret = MatchTensorizeBody(stage->op.as<ComputeOpNode>(),
                              stage,
                              {{}},
                              as_unordered_map(out_dom),
                              as_unordered_map(in_region),
                              intrin,
                              &vrange);
  });
}  // namespace te
}  // namespace tvm
