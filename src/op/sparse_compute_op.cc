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
 *  Copyright (c) 2017 by Contributors
 * \brief Compute Op.
 * \file compute_op.cc
 */
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include <string>
#include <utility>
#include "compute_op.h"
#include "op_util.h"
#include "../schedule/message_passing.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {

using namespace ir;

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<SparseComputeOpNode>([](const SparseComputeOpNode *op, IRPrinter *p) {
    p->stream << "sparse_compute(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(SparseComputeOpNode);

int SparseComputeOpNode::num_outputs() const {
  return body.size();
}

Type SparseComputeOpNode::output_dtype(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  return body[idx].type();
}

SparseFormat SparseComputeOpNode::output_sformat(size_t idx) const {
  return sformat;
}

Operation SparseComputeOpNode::make(std::string name,
                                    std::string tag,
                                    SparseFormat sformat,
                                    Map<std::string, NodeRef> attrs,
                                    Array<IterVar> axis,
                                    Array<Expr> body) {
  if (!attrs.defined()) {
    attrs = Map<std::string, NodeRef>();
  }
  auto n = make_node<SparseComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->sformat = std::move(sformat);
  n->attrs = std::move(attrs);
  n->axis = std::move(axis);
  n->body = std::move(body);
  if (n->body[0]->is_type<ir::Reduce>()) {
    const ir::Reduce* reduce = n->body[0].as<ir::Reduce>();
    n->reduce_axis = reduce->axis;
  }
  // VerifyComputeOp(n.get());
  return Operation(n);
}

// The schedule related logics
Array<Tensor> SparseComputeOpNode::InputTensors() const {
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

Operation SparseComputeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  // VerifyComputeOp(this);
  Array<Expr> arr;
  if (this->body[0]->is_type<ir::Reduce>()) {
    // Specially handle reduce so the replaced op
    // still share all the components
    Expr new_reduce = op::ReplaceTensor(this->body[0], rmap);
    if (!new_reduce.same_as(this->body[0])) {
      const ir::Reduce* r = new_reduce.as<ir::Reduce>();
      for (size_t k = 0; k < this->body.size(); ++k) {
        auto n = make_node<ir::Reduce>(*r);
        n->value_index = static_cast<int>(k);
        n->type = r->source[k].type();
        arr.push_back(Expr(n));
      }
    } else {
      arr = this->body;
    }
  } else {
    arr = UpdateArray(this->body, [&rmap] (const Expr& e) {
        return op::ReplaceTensor(e, rmap);
      });
  }
  if (!arr.same_as(this->body)) {
    return SparseComputeOpNode::make(
        this->name, this->tag, this->sformat, this->attrs, this->axis, arr);
  } else {
    return self;
  }
}

void SparseComputeOpNode::PropBoundToInputs(
    const Operation& self,
    arith::Analyzer* analyzer,
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

size_t SparseComputeOpNode::num_schedulable_dims() const {
  return axis.size();
}

// Build a reduction body.
void MakeReduction(const SparseComputeOpNode* op,
                   const Array<Tensor>& tensors,
                   Stmt* init,
                   Stmt* provide) {
  Array<Expr>  args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  std::vector<Stmt> inits, provides;

  size_t size = op->body.size();
  const Reduce* reduce = op->body[0].as<Reduce>();
  CHECK(reduce);
  const CommReducerNode* combiner = reduce->combiner.as<CommReducerNode>();
  CHECK(combiner);
  Array<Expr> lhs;
  for (size_t i = 0; i < size; ++i) {
    lhs.push_back(tensors[i](args));
  }
  Array<Expr> init_value = combiner->identity_element;
  Array<Expr> update_value = (*combiner)(lhs, reduce->source);
  for (size_t i = 0; i < size; ++i) {
    Tensor t = tensors[i];
    inits.emplace_back(Provide::make(
          t->op, t->value_index, init_value[i], args));
    provides.emplace_back(Provide::make(
          t->op, t->value_index, update_value[i], args));
  }
  *init = Block::make(inits);
  *provide = Block::make(provides);
  if (!is_one(reduce->condition)) {
    *provide = IfThenElse::make(reduce->condition, *provide);
  }
}

// Normal computation.
Stmt MakeProvide(const SparseComputeOpNode* op,
                 const Tensor& t) {
  Array<Expr> args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  return Provide::make(t->op, t->value_index, op->body[t->value_index], args);
}

inline bool IsSparseTensor(Tensor t) {
  for (auto ty : t->sformat->types) {
    auto i = ty.as<IntImm>();
    CHECK(i);
    if (i->value != kDense) {
        return true;
    }
  }
  return false;
}


class SparseRewriter : public IRMutator {
 public:
  explicit SparseRewriter(Tensor origin,
                          Tensor val,
                          Tensor crd,
                          Expr idx)
      : origin_(origin), val_(val), crd_(crd), idx_(idx) {}

  Expr Mutate_(const ir::Call* op, const Expr& e) {
    if (op->call_type == ir::Call::Halide) {
      Tensor t = Operation(op->func.node_).output(op->value_index);
      if (t == origin_) {
        for (size_t i = 0; i < t->sformat->types.size(); ++i) {
          Expr stype = t->sformat->types[i];
          CHECK(stype.as<IntImm>());
          if (stype.as<IntImm>()->value == kSparse) {
            if (op->args[i].as<Variable>()) {
              const auto* n = op->args[i].as<Variable>();
              crd_map[n] = crd_(idx_);
            }
          }
        }
        // Expr ret = ir::Call::make(
        //     op->type, it->second->op->name, op->args,
        //     op->call_type, it->second->op, it->second->value_index);
        Expr ret = val_(idx_);
        found = true;
        return IRMutator::Mutate_(ret.as<ir::Call>(), ret);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  // whether it is found.
  bool found{false};
  std::unordered_map<const Variable*, Expr> crd_map;

 private:
  Tensor origin_;
  Tensor val_;
  Tensor crd_;
  Expr idx_;
};

Stmt RewriteSparseTensor(Stmt stmt,
                         Tensor origin,
                         Tensor val,
                         Tensor crd,
                         Expr idx,
                         std::unordered_map<const Variable*, Expr>* crd_map) {
  SparseRewriter repl(origin, val, crd, idx);
  Stmt ret = repl.Mutate(stmt);
  *crd_map = repl.crd_map;
  return repl.found ? ret : stmt;
}


Stmt MakeComputeStmt(const SparseComputeOpNode* self,
                     const Stage& stage,
                     const std::unordered_map<IterVar, Range>& dom_map,
                     bool debug_keep_trivial_loop) {
  // grab the nest structure
  ComputeLoopNest n = ComputeLoopNest::make(self, stage, dom_map, debug_keep_trivial_loop);
  // Normal loop structure
  n.init_nest.emplace_back(op::MakeIfNest(n.init_predicates));
  n.main_nest.emplace_back(op::MakeIfNest(n.main_predicates));

  // prepare auxiliary tensors
  Tensor t = self->InputTensors()[0];
  CHECK (IsSparseTensor(t));
  SparseFormat sformat = t->sformat;
  Tensor t_val = placeholder({Var("s0")}, t->dtype, "A_val");
  Tensor t_pos = placeholder({Var("s1")}, Int(32), "A1_pos");
  Tensor t_crd = placeholder({Var("s2")}, Int(32), "A1_crd");

  // final var to index sparse tensor
  Var idx = Var("idx", Int(32));

  if (self->reduce_axis.size() != 0) {
    // make reduction.
    Stmt init, provide;
    Array<Tensor> source;
    for (size_t i = 0; i < self->body.size(); ++i) {
      source.push_back(stage->op.output(i));
    }
    MakeReduction(self, source, &init, &provide);

    // rewrite tensor for sparse
    // map <Tensor, Expr> A(...) -> A_val(k)
    // map <Var, Var> k->crd
    std::unordered_map<const Variable*, Expr> crd_map;
    provide = RewriteSparseTensor(provide, t, t_val, t_crd, idx, &crd_map);
    provide = Substitute(provide, crd_map);


    init = MergeNest(n.init_nest, init);
    init = op::Substitute(init, n.init_vmap);
    // common nest
    std::vector<std::vector<Stmt> > common(
        n.main_nest.begin(), n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt> > reduce(
        n.main_nest.begin() + n.num_common_loop + 1, n.main_nest.end());

    // Construct reduce loop for sparse
    std::vector<std::vector<Stmt> > sparse_reduce;
    IterVar ivar = self->axis[0];

    Expr begin = t_pos({ivar});
    Expr end = t_pos({ivar + 1});
    Expr extent = end - begin;

    Stmt no_op = Evaluate::make(0);
    Var var = self->reduce_axis[0]->var;
    Stmt s = For::make(idx,
                       begin, extent,
                       ForType::Serial, DeviceAPI::None, no_op);
    sparse_reduce.push_back({s});
    sparse_reduce[0].emplace_back(
        AttrStmt::make(idx, ir::attr::loop_scope, idx, no_op));

    provide = MergeNest(sparse_reduce, provide);
    if (debug_keep_trivial_loop) {
      provide = MergeNest(common, provide);
    } else {
      provide = MergeNest(common, Block::make(init, provide));
    }
    // run substitution in the on the full nest, because  loop condition
    // could depend on outer loops.
    return op::Substitute(provide, n.main_vmap);
  } else {
    std::vector<Stmt> provides;
    for (size_t i = 0; i < self->body.size(); ++i) {
      provides.emplace_back(MakeProvide(self, stage->op.output(i)));
    }
    Stmt provide = Block::make(provides);
    provide = MergeNest(n.main_nest, provide);
    // run substitution in the on the full nest, because  loop condition
    // could depend on outer loops.
    return op::Substitute(provide, n.main_vmap);
  }
}

enum class ComputeType {
  kNormal,
  kCrossThreadReduction,
  kTensorize
};

ComputeType DetectComputeType(const SparseComputeOpNode* self,
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
  CHECK(normal_red == 0 || thread_red == 0)
      << "Cannot mix normal reduction with thread reduce";
  if (thread_red != 0) {
    return ComputeType::kCrossThreadReduction;
  } else {
    return ComputeType::kNormal;
  }
}

// implement the provide utility.
Stmt SparseComputeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  ComputeType ctype = DetectComputeType(this, stage);
  CHECK(ctype == ComputeType::kNormal);
  return MakeComputeStmt(this, stage, dom_map, debug_keep_trivial_loop);
}

}  // namespace tvm
