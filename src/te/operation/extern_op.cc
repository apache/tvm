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

DataType ExternOpNode::output_dtype(size_t i) const { return output_placeholders[i]->dtype; }

Array<PrimExpr> ExternOpNode::output_shape(size_t i) const { return output_placeholders[i]->shape; }

ExternOp::ExternOp(std::string name, std::string tag, Map<String, ffi::Any> attrs,
                   Array<Tensor> inputs, Array<Buffer> input_placeholders,
                   Array<Buffer> output_placeholders, Stmt body) {
  if (!attrs.defined()) {
    attrs = Map<String, ffi::Any>();
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
    .set_body_typed([](std::string name, std::string tag, Map<String, ffi::Any> attrs,
                       Array<Tensor> inputs, Array<Buffer> input_placeholders,
                       Array<Buffer> output_placeholders, Stmt body) {
      return ExternOp(name, tag, attrs, inputs, input_placeholders, output_placeholders, body);
    });

Array<Tensor> ExternOpNode::InputTensors() const { return inputs; }

}  // namespace te
}  // namespace tvm
