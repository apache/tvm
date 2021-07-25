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
 * \brief External computation rule.
 * \file extern_op.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include <unordered_set>

#include "op_utils.h"

namespace tvm {
namespace te {
using namespace tir;
// ExternOpNode
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ExternOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ExternOpNode*>(node.get());
      p->stream << "extern(" << op->name << ", " << op << ")";
    });

TVM_REGISTER_NODE_TYPE(ExternOpNode);

int ExternOpNode::num_outputs() const { return static_cast<int>(output_placeholders.size()); }

Array<IterVar> ExternOpNode::root_iter_vars() const { return {}; }

DataType ExternOpNode::output_dtype(size_t i) const { return output_placeholders[i]->dtype; }

Array<PrimExpr> ExternOpNode::output_shape(size_t i) const { return output_placeholders[i]->shape; }

ExternOp::ExternOp(std::string name, std::string tag, Map<String, ObjectRef> attrs,
                   Array<Tensor> inputs, Array<Buffer> input_placeholders,
                   Array<Buffer> output_placeholders, Stmt body) {
  if (!attrs.defined()) {
    attrs = Map<String, ObjectRef>();
  }
  auto n = make_object<ExternOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  ICHECK_EQ(inputs.size(), input_placeholders.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    ICHECK_EQ(inputs[i]->dtype, input_placeholders[i]->dtype);
    ICHECK_EQ(inputs[i]->shape.size(), input_placeholders[i]->shape.size());
    for (size_t dim = 0; dim < inputs[i]->shape.size(); ++dim) {
      ICHECK(inputs[i]->shape[dim].same_as(input_placeholders[i]->shape[dim]));
    }
    ICHECK_EQ(input_placeholders[i]->strides.size(), 0U);
  }
  n->inputs = std::move(inputs);
  n->input_placeholders = std::move(input_placeholders);
  n->output_placeholders = std::move(output_placeholders);
  n->body = std::move(body);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("te.ExternOp")
    .set_body_typed([](std::string name, std::string tag, Map<String, ObjectRef> attrs,
                       Array<Tensor> inputs, Array<Buffer> input_placeholders,
                       Array<Buffer> output_placeholders, Stmt body) {
      return ExternOp(name, tag, attrs, inputs, input_placeholders, output_placeholders, body);
    });

Array<Tensor> ExternOpNode::InputTensors() const { return inputs; }

Operation ExternOpNode::ReplaceInputs(const Operation& self,
                                      const std::unordered_map<Tensor, Tensor>& rmap) const {
  ICHECK_EQ(self.operator->(), this);
  auto n = make_object<ExternOpNode>(*this);
  n->body = ReplaceTensor(this->body, rmap);
  for (size_t i = 0; i < n->inputs.size(); ++i) {
    Tensor t = n->inputs[i];
    if (rmap.count(t)) {
      n->inputs.Set(i, rmap.at(t));
    }
  }

  if (body.same_as(n->body) && inputs.same_as(n->inputs)) {
    return self;
  } else {
    return Operation(n);
  }
}

void ExternOpNode::PropBoundToInputs(const Operation& self, arith::Analyzer* analyzer,
                                     const std::unordered_map<const VarNode*, IntSet>& dom_map,
                                     std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  for (Tensor t : this->inputs) {
    auto it = out_dom_map->find(t);
    if (it == out_dom_map->end()) continue;
    TensorDom& dom = it->second;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      dom.data[i].emplace_back(
          IntSet::FromRange(Range::FromMinExtent(make_const(t->shape[i].dtype(), 0), t->shape[i])));
    }
  }
}

void ExternOpNode::GatherBound(const Operation& self,
                               const std::unordered_map<Tensor, TensorDom>& tensor_dom,
                               std::unordered_map<IterVar, Range>* out_dom_map) const {}

Stmt ExternOpNode::BuildRealize(const Stage& stage,
                                const std::unordered_map<IterVar, Range>& realize_map,
                                const Stmt& body, String storage_scope) const {
  ICHECK_EQ(stage->op.get(), this);
  Stmt realize_body = body;
  for (int k = 0; k < num_outputs(); ++k) {
    Tensor t = stage->op.output(k);
    Region bounds;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      bounds.push_back(Range::FromMinExtent(make_const(t->shape[i].dtype(), 0), t->shape[i]));
    }
    realize_body = tir::ProducerRealize(t, bounds, const_true(), realize_body, storage_scope);
  }
  return realize_body;
}

Stmt ExternOpNode::BuildProvide(const Stage& stage,
                                const std::unordered_map<IterVar, Range>& dom_map,
                                bool debug_keep_trivial_loop) const {
  ICHECK_EQ(stage->op.operator->(), this);
  Stmt ret = AttrStmt(make_zero(DataType::Int(32)), tir::attr::extern_scope, 0, this->body);
  auto f_push_bind = [&ret](Buffer buffer, Tensor tensor) {
    Array<ObjectRef> bind_spec;
    Array<PrimExpr> tuple;
    bind_spec.push_back(buffer);
    bind_spec.push_back(tensor);
    for (size_t k = 0; k < buffer->shape.size(); ++k) {
      tuple.push_back(make_const(buffer->shape[k].dtype(), 0));
      tuple.push_back(buffer->shape[k]);
    }
    ret = AttrStmt(bind_spec, tir::attr::buffer_bind_scope,
                   Call(DataType::Handle(), builtin::tvm_tuple(), tuple), ret);
  };
  for (size_t i = output_placeholders.size(); i != 0; --i) {
    f_push_bind(output_placeholders[i - 1], stage->op.output(i - 1));
  }
  for (size_t i = inputs.size(); i != 0; --i) {
    f_push_bind(input_placeholders[i - 1], inputs[i - 1]);
  }
  return ret;
}
}  // namespace te
}  // namespace tvm
