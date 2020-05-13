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
 * \file src/ir/tensor_type.cc
 * \brief The type system AST nodes of Relay.
 */
#include <tvm/ir/tensor_type.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>

namespace tvm {

using tvm::ReprPrinter;
using namespace tvm::runtime;

TensorType::TensorType(Array<PrimExpr> shape, DataType dtype) {
  ObjectPtr<TensorTypeNode> n = make_object<TensorTypeNode>();
  n->shape = std::move(shape);
  n->dtype = std::move(dtype);
  data_ = std::move(n);
}

TensorType TensorType::Scalar(DataType dtype) { return TensorType({}, dtype); }

PrimExpr TensorTypeNode::Size() const {
  if (shape.size() == 0) {
    return tir::make_const(DataType::Int(64), 1);
  }

  PrimExpr size = shape[0];
  for (size_t i = 1; i < shape.size(); ++i) {
    size *= shape[i];
  }
  return size;
}

TVM_REGISTER_NODE_TYPE(TensorTypeNode);

TVM_REGISTER_GLOBAL("ir.TensorType").set_body_typed([](Array<PrimExpr> shape, DataType dtype) {
  return TensorType(shape, dtype);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TensorTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TensorTypeNode*>(ref.get());
      p->stream << "TensorType(" << node->shape << ", " << node->dtype << ")";
    });

}  // namespace tvm
