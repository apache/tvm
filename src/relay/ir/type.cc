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
 * \file src/tvm/ir/type.cc
 * \brief The type system AST nodes of Relay.
 */
#include <tvm/relay/type.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relay {

using tvm::NodePrinter;
using namespace tvm::runtime;

TensorType TensorTypeNode::make(Array<IndexExpr> shape, DataType dtype) {
  ObjectPtr<TensorTypeNode> n = make_object<TensorTypeNode>();
  n->shape = std::move(shape);
  n->dtype = std::move(dtype);
  return TensorType(n);
}

TensorType TensorTypeNode::Scalar(DataType dtype) {
  return TensorTypeNode::make({}, dtype);
}

IndexExpr TensorTypeNode::Size() const {
  if (shape.size() == 0) {
    return tir::make_const(DataType::Int(64), 1);
  }

  IndexExpr size = shape[0];
  for (size_t i = 1; i < shape.size(); ++i) {
    size *= shape[i];
  }
  return size;
}

TVM_REGISTER_NODE_TYPE(TensorTypeNode);

TVM_REGISTER_GLOBAL("relay._make.TensorType")
.set_body_typed(TensorTypeNode::make);

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<TensorTypeNode>([](const ObjectRef& ref, NodePrinter* p) {
  auto* node = static_cast<const TensorTypeNode*>(ref.get());
  p->stream << "TensorType(" << node->shape << ", " << node->dtype << ")";
});

IncompleteType IncompleteTypeNode::make(Kind kind) {
  auto n = make_object<IncompleteTypeNode>();
  n->kind = std::move(kind);
  return IncompleteType(n);
}

TVM_REGISTER_NODE_TYPE(IncompleteTypeNode);

TVM_REGISTER_GLOBAL("relay._make.IncompleteType")
.set_body_typed([](int kind) {
    return IncompleteTypeNode::make(static_cast<Kind>(kind));
  });

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<IncompleteTypeNode>([](const ObjectRef& ref, NodePrinter* p) {
    auto* node = static_cast<const IncompleteTypeNode*>(ref.get());
    p->stream << "IncompleteTypeNode(" << node->kind << ", " << node << ")";
  });


RefType RefTypeNode::make(Type value) {
  ObjectPtr<RefTypeNode> n = make_object<RefTypeNode>();
  n->value = std::move(value);
  return RefType(n);
}

TVM_REGISTER_GLOBAL("relay._make.RefType")
.set_body_typed(RefTypeNode::make);

TVM_REGISTER_NODE_TYPE(RefTypeNode);

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<RefTypeNode>([](const ObjectRef& ref, NodePrinter* p) {
  auto* node = static_cast<const RefTypeNode*>(ref.get());
  p->stream << "RefTypeNode(" << node->value << ")";
});

TVM_REGISTER_GLOBAL("relay._make.Any")
.set_body_typed([]() { return Any::make(); });

}  // namespace relay
}  // namespace tvm
