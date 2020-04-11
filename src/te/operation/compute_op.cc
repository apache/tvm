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
 * \brief Compute Op.
 * \file compute_op.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <unordered_set>
#include <string>
#include <utility>
#include "compute_op.h"
#include "op_util.h"
#include "../schedule/message_passing.h"
#include "../../arith/compute_expr.h"
#include "../../arith/interval_set.h"

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ComputeOpNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const ComputeOpNode*>(node.get());
    p->stream << "compute(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

/// Verify if ComputeOp is valid with respect to Reduce operations.
static void VerifyComputeOp(const ComputeOpNode *op);

inline bool ReduceEqual(const tir::ReduceNode* a, const tir::ReduceNode* b) {
  return (a->combiner.same_as(b->combiner)) &&
         (a->source.same_as(b->source)) &&
         (a->axis.same_as(b->axis)) &&
         (a->condition.same_as(b->condition));
}

int ComputeOpNode::num_outputs() const {
  return body.size();
}

Array<IterVar> BaseComputeOpNode::root_iter_vars() const {
  if (reduce_axis.size() == 0) return axis;
  Array<IterVar> ret = axis;
  for (IterVar iv : reduce_axis) {
    ret.push_back(iv);
  }
  return ret;
}

DataType ComputeOpNode::output_dtype(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  return body[idx].dtype();
}

Array<PrimExpr> BaseComputeOpNode::output_shape(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  // for now, all outputs of a BaseComputeOp have the same shape
  Array<PrimExpr> shape;
  for (const auto& ivar : this->axis) {
    const Range& r = ivar->dom;
    shape.push_back(r->extent);
  }
  return shape;
}

Tensor compute(Array<PrimExpr> shape,
               FCompute fcompute,
               std::string name,
               std::string tag,
               Map<std::string, ObjectRef> attrs) {
  auto op_node = make_object<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVarNode::make(
        Range(0, shape[i]), Var(os.str(), shape[i].dtype()), kDataPar));
    args.push_back(axis.back()->var);
  }

  return ComputeOpNode::make(
      name, tag, attrs, axis, {fcompute(args)}).output(0);
}

Array<Tensor> compute(Array<PrimExpr> shape,
                      FBatchCompute fcompute,
                      std::string name,
                      std::string tag,
                      Map<std::string, ObjectRef> attrs) {
  auto op_node = make_object<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVarNode::make(
        Range(0, shape[i]), Var(os.str(), shape[i].dtype()), kDataPar));
    args.push_back(axis.back()->var);
  }

  Operation op = ComputeOpNode::make(name, tag, attrs, axis, fcompute(args));
  Array<Tensor> outputs;
  for (int idx = 0; idx < op->num_outputs(); ++idx) {
    outputs.push_back(op.output(idx));
  }
  return outputs;
}

Operation ComputeOpNode::make(std::string name,
                              std::string tag,
                              Map<std::string, ObjectRef> attrs,
                              Array<IterVar> axis,
                              Array<PrimExpr> body) {
  if (!attrs.defined()) {
    attrs = Map<std::string, ObjectRef>();
  }
  auto n = make_object<ComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->axis = std::move(axis);
  n->body = std::move(body);
  if (n->body[0]->IsInstance<tir::ReduceNode>()) {
    const tir::ReduceNode* reduce = n->body[0].as<tir::ReduceNode>();
    n->reduce_axis = reduce->axis;
  }
  VerifyComputeOp(n.get());
  return Operation(n);
}

TVM_REGISTER_GLOBAL("te.ComputeOp")
.set_body_typed(ComputeOpNode::make);


// The schedule related logics
Array<Tensor> ComputeOpNode::InputTensors() const {
  Array<Tensor> ret;
  std::unordered_set<Tensor> visited;
  for (auto& e : body) {
    tir::PostOrderVisit(e, [&ret, &visited](const ObjectRef& n) {
        const tir::CallNode *call = n.as<tir::CallNode>();
        if (call != nullptr && call->func.defined()) {
          Tensor t = Downcast<Operation>(call->func).output(call->value_index);
          if (!visited.count(t)) {
            ret.push_back(t);
            visited.insert(t);
          }
        }
      });
  }
  return ret;
}

Operation ComputeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  VerifyComputeOp(this);
  Array<PrimExpr> arr;
  if (this->body[0]->IsInstance<tir::ReduceNode>()) {
    // Specially handle reduce so the replaced op
    // still share all the components
    PrimExpr new_reduce = te::ReplaceTensor(this->body[0], rmap);
    if (!new_reduce.same_as(this->body[0])) {
      const tir::ReduceNode* r = new_reduce.as<tir::ReduceNode>();
      for (size_t k = 0; k < this->body.size(); ++k) {
        auto n = make_object<tir::ReduceNode>(*r);
        n->value_index = static_cast<int>(k);
        n->dtype = r->source[k].dtype();
        arr.push_back(PrimExpr(n));
      }
    } else {
      arr = this->body;
    }
  } else {
    arr = UpdateArray(this->body, [&rmap] (const PrimExpr& e) {
        return te::ReplaceTensor(e, rmap);
      });
  }
  if (!arr.same_as(this->body)) {
    return ComputeOpNode::make(
        this->name, this->tag, this->attrs, this->axis, arr);
  } else {
    return self;
  }
}

void ComputeOpNode::PropBoundToInputs(
    const Operation& self,
    arith::Analyzer* analyzer,
    const std::unordered_map<const VarNode*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  auto fvisit = [&dom_map, out_dom_map, analyzer](const ObjectRef& n) {
    auto *call = n.as<tir::CallNode>();
    if (call != nullptr && call->func.defined()) {
      Tensor t = Downcast<Operation>(call->func).output(call->value_index);
      if (t->op.defined() && out_dom_map->count(t)) {
        TensorDom& dom = out_dom_map->at(t);
        for (size_t i = 0; i < t.ndim(); ++i) {
          // We assume that the value of the argument cannot be out of bounds (otherwise it is
          // undefined behaviour), so we can intersect the estimated set of the argument with the
          // range expected by the tensor. However, intersection may result in overly complex
          // expressions, so we perform a more relaxed form of intersection.
          IntSet arg_intset = EvalSet(call->args[i], dom_map);
          const arith::IntervalSetNode* arg_interval = arg_intset.as<arith::IntervalSetNode>();
          if (arg_interval) {
            PrimExpr shape_i_min_value = make_zero(t->shape[i].dtype());
            PrimExpr shape_i_max_value = t->shape[i] - 1;
            PrimExpr min_value = arg_interval->min_value;
            PrimExpr max_value = arg_interval->max_value;
            // Prefer the shape bounds only when we can prove they are tighter.
            if (arith::is_neg_inf(min_value) ||
                analyzer->CanProve(shape_i_min_value >= min_value)) {
              min_value = shape_i_min_value;
            }
            if (arith::is_pos_inf(max_value) ||
                analyzer->CanProve(shape_i_max_value <= max_value)) {
              max_value = shape_i_max_value;
            }
            dom.data[i].push_back(IntSet::interval(min_value, max_value));
          } else {
            dom.data[i].push_back(arg_intset);
          }
        }
      }
    }
  };
  for (auto& e : body) tir::PostOrderVisit(e, fvisit);
}

void BaseComputeOpNode::GatherBound(
    const Operation& self,
    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
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

Stmt BaseComputeOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& realize_map,
    const Stmt& body) const {
  CHECK_EQ(stage->op.get(), this);
  Region bounds;
  for (IterVar iv : this->axis) {
    bounds.push_back(realize_map.at(iv));
  }
  Stmt realize = body;
  for (int i = this->num_outputs(); i > 0; --i) {
    Tensor t = stage->op.output(i-1);
    realize = tir::RealizeNode::make(t->op, t->value_index,
      t->dtype, bounds, const_true(), realize);
    // alignment requirement, only useful for compute
    for (size_t i = 0; i < num_schedulable_dims(); ++i) {
      auto it = stage->iter_var_attrs.find(this->axis[i]);
      if (it != stage->iter_var_attrs.end()) {
        IterVarAttr attr = (*it).second;
        if (attr->dim_align_factor != 0) {
          Array<PrimExpr> tuple = {static_cast<int>(i),
                               attr->dim_align_factor,
                               attr->dim_align_offset};
          realize = tir::AttrStmtNode::make(
              t, tir::attr::buffer_dim_align,
              CallNode::make(DataType::Handle(),
                             tir::intrinsic::tvm_tuple,
                             tuple, CallNode::Intrinsic),
              realize);
        }
      }
    }
  }
  return realize;
}

size_t ComputeOpNode::num_schedulable_dims() const {
  return axis.size();
}

// Build a reduction body.
void MakeReduction(const ComputeOpNode* op,
                   const Array<Tensor>& tensors,
                   Stmt* init,
                   Stmt* provide) {
  Array<PrimExpr>  args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  std::vector<Stmt> inits, provides;

  size_t size = op->body.size();
  const ReduceNode* reduce = op->body[0].as<ReduceNode>();
  CHECK(reduce);
  const CommReducerNode* combiner = reduce->combiner.as<CommReducerNode>();
  CHECK(combiner);
  Array<PrimExpr> lhs;
  for (size_t i = 0; i < size; ++i) {
    lhs.push_back(tensors[i](args));
  }
  Array<PrimExpr> init_value = combiner->identity_element;
  Array<PrimExpr> update_value = (*combiner)(lhs, reduce->source);
  for (size_t i = 0; i < size; ++i) {
    Tensor t = tensors[i];
    inits.emplace_back(ProvideNode::make(
          t->op, t->value_index, init_value[i], args));
    provides.emplace_back(ProvideNode::make(
          t->op, t->value_index, update_value[i], args));
  }
  *init = SeqStmt::Flatten(inits);
  *provide = SeqStmt::Flatten(provides);
  if (!is_one(reduce->condition)) {
    *provide = IfThenElseNode::make(reduce->condition, *provide);
  }
}

// Normal computation.
Stmt MakeProvide(const ComputeOpNode* op,
                 const Tensor& t) {
  Array<PrimExpr> args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  return ProvideNode::make(t->op, t->value_index, op->body[t->value_index], args);
}

Stmt MakeComputeStmt(const ComputeOpNode* self,
                     const Stage& stage,
                     const std::unordered_map<IterVar, Range>& dom_map,
                     bool debug_keep_trivial_loop) {
  // grab the nest structure
  ComputeLoopNest n = ComputeLoopNest::make(self, stage, dom_map, debug_keep_trivial_loop);
  // Normal loop structure
  n.init_nest.emplace_back(MakeIfNest(n.init_predicates));
  n.main_nest.emplace_back(MakeIfNest(n.main_predicates));
  if (self->reduce_axis.size() != 0) {
    // make reduction.
    Stmt init, provide;
    Array<Tensor> source;
    for (size_t i = 0; i < self->body.size(); ++i) {
      source.push_back(stage->op.output(i));
    }
    MakeReduction(self, source, &init, &provide);
    init = MergeNest(n.init_nest, init);
    init = Substitute(init, n.init_vmap);
    // common nest
    std::vector<std::vector<Stmt> > common(
        n.main_nest.begin(), n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt> > reduce(
        n.main_nest.begin() + n.num_common_loop + 1, n.main_nest.end());
    provide = MergeNest(reduce, provide);
    if (debug_keep_trivial_loop) {
      provide = MergeNest(common, provide);
    } else {
      provide = MergeNest(common, SeqStmt::Flatten(init, provide));
    }
    // run substitution in the on the full nest, because  loop condition
    // could depend on outer loops.
    return Substitute(provide, n.main_vmap);
  } else {
    std::vector<Stmt> provides;
    for (size_t i = 0; i < self->body.size(); ++i) {
      provides.emplace_back(MakeProvide(self, stage->op.output(i)));
    }
    Stmt provide = SeqStmt::Flatten(provides);
    provide = MergeNest(n.main_nest, provide);
    // run substitution in the on the full nest, because  loop condition
    // could depend on outer loops.
    return Substitute(provide, n.main_vmap);
  }
}

enum class ComputeType {
  kNormal,
  kCrossThreadReduction,
  kTensorize
};

ComputeType DetectComputeType(const ComputeOpNode* self,
                              const Stage& stage) {
  // Verify correctness of leaf nest.
  int normal_red = 0, thread_red = 0, tensorize = 0;

  for (IterVar iv : stage->leaf_iter_vars) {
    IterVarAttr attr;
    auto it = stage->iter_var_attrs.find(iv);
    if (it != stage->iter_var_attrs.end()) {
      attr = (*it).second;
    }
    if (attr.defined() && attr->iter_type == kTensorized) {
      ++tensorize;
    }
    if (iv->iter_type == kCommReduce) {
      if (attr.defined() && attr->bind_thread.defined()) {
        ++thread_red;
      } else {
        ++normal_red;
      }
    } else {
      CHECK_EQ(thread_red, 0)
          << "Cross thread reduce cannot swap with normal data axis";
    }
  }
  if (tensorize != 0) {
    CHECK(thread_red == 0)
        << "Cannot mix cross thread reduction with Tensorize";
    return ComputeType::kTensorize;
  }
  if (thread_red != 0) {
    return ComputeType::kCrossThreadReduction;
  } else {
    return ComputeType::kNormal;
  }
}

// implement the provide utility.
Stmt ComputeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  ComputeType ctype = DetectComputeType(this, stage);
  if (ctype == ComputeType::kCrossThreadReduction) {
    // specially handle cross thread reduction.
    return MakeCrossThreadReduction(this, stage, dom_map, debug_keep_trivial_loop);
  } else if (ctype == ComputeType::kTensorize) {
    return MakeTensorize(this, stage, dom_map, debug_keep_trivial_loop);
  } else {
    return MakeComputeStmt(this, stage, dom_map, debug_keep_trivial_loop);
  }
}

ComputeLoopNest ComputeLoopNest::make(
    const BaseComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) {
  CHECK_EQ(stage->op.operator->(), self);
  ComputeLoopNest ret;
  // make main loop nest
  ret.main_nest = MakeLoopNest(
      stage, dom_map, 0, false, std::unordered_set<IterVar>(), &ret.main_vmap,
      debug_keep_trivial_loop);
  ret.main_predicates = MakeBoundCheck(
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
    for (size_t i = 0; i < self->num_schedulable_dims(); ++i) {
      update_state[self->axis[i]] = 1;
    }
    // find which iter var is related to reduction and which is related to axis.
    te::PassDownBitMaskOr(stage, &update_state);
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
    // skip loops that are related to reduction and are unrelated to axis.
    std::unordered_set<IterVar> skip_iter;
    for (auto kv : update_state) {
      int flag = kv.second;
      if (flag == 2) skip_iter.insert(kv.first);
    }
    ret.init_nest = MakeLoopNest(
        stage, dom_map, begin_loop, true,
        skip_iter, &(ret.init_vmap), debug_keep_trivial_loop);
    ret.init_predicates = MakeBoundCheck(
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

namespace {
/*!
 * \brief Verify if ComputeOp is valid with respect to Reduce operations.
 *
 *  The following two properties are verified:
 *  (1) All Reduce operations must exist at top level.
 *  (2) For a list of operations, if one is Reduce, then the others
 *      must be Reduce as well; and their inputs should have the
 *      same attribute except value_index.
 */
class ComputeVerifier final : protected tir::ExprVisitor {
 public:
  /// Special member functions
  //@{
  explicit ComputeVerifier(const ComputeOpNode* compute)
      : compute_(compute), reduce_(compute->body[0].as<tir::ReduceNode>()) {}
  virtual ~ComputeVerifier() = default;
  ComputeVerifier(const ComputeVerifier&) = delete;
  ComputeVerifier(ComputeVerifier&&) = delete;
  ComputeVerifier& operator=(const ComputeVerifier&) = delete;
  ComputeVerifier& operator=(ComputeVerifier&&) = delete;
  //@}

  /// Interface to perform compute verification
  void Run() {
    for (const PrimExpr e : compute_->body) {
      // Check for consistency of top level reductions
      const tir::ReduceNode* reduce = e.as<tir::ReduceNode>();
      CHECK((reduce && reduce_) || (!reduce && !reduce_))
          << "All ComputeOp should be consistent "
          << "with being Reduce operation or not.";

      if (reduce && reduce_) {
        CHECK(ReduceEqual(reduce, reduce_))
            << "The Reduce inputs of ComputeOp should "
            << "have the same attribute except value_index";
      }

      level_ = 0;
      ExprVisitor::VisitExpr(e);
    }
  }

 protected:
  /// Visitor implementation
  //@{
  void VisitExpr(const PrimExpr& n) final {
    ++level_;
    ExprVisitor::VisitExpr(n);
    --level_;
  }

  void VisitExpr_(const tir::ReduceNode* op) final {
    // Check for non top level reductions
    CHECK(0 == level_)
        << "Reductions are only allowed at the top level of compute. "
        << "Please create another tensor for further composition.";
  }
  //@}

 private:
  const ComputeOpNode* compute_{nullptr};  ///< ComputeOpNode to verify
  const tir::ReduceNode* reduce_{nullptr};      ///< Top level Reduce operation
  int level_{0};                           ///< Level of op being processed
};
}  // namespace

/// Verify if ComputeOp is valid with respect to Reduce operations.
static void VerifyComputeOp(const ComputeOpNode* op) {
  ComputeVerifier v(op);
  v.Run();
}

Stmt TransformUpdate(const Stage& stage,
                     const std::unordered_map<IterVar, Range>& dom_map,
                     const ComputeLoopNest& n,
                     Stmt body,
                     Stmt update) {
  Array<PrimExpr> conds;
  std::unordered_set<const VarNode*> banned;
  for (size_t i = 0; i < stage->leaf_iter_vars.size(); ++i) {
    IterVar iv = stage->leaf_iter_vars[i];
    auto iit = stage->iter_var_attrs.find(iv);
    if (iit != stage->iter_var_attrs.end()) {
      const IterVarAttr& attr = (*iit).second;
      if (attr->iter_type == kTensorized) {
        break;
      }
    }
    if (iv->iter_type == kCommReduce) {
      auto vit = dom_map.find(iv);
      CHECK(vit != dom_map.end());
      const Range& vrange = vit->second;
      conds.push_back(likely(iv->var > vrange->min));
      banned.insert(iv->var.get());
    }
  }
  for (const PrimExpr& pred : n.main_predicates) {
    if (tir::ExprUseVar(pred, banned)) {
      LOG(FATAL) << "Tensorize update transform failed, the condition "
                 << pred << " has a conflict with the reset condition";
    }
  }

  return IfThenElseNode::make(arith::ComputeReduce<tir::OrNode>(conds, const_true(1)),
                          update, body);
}

}  // namespace te
}  // namespace tvm
