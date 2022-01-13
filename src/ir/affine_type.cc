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
 * \file src/ir/affine_type.cc
 * \brief The Type information for quantized nodes.
 */
#include <tvm/ir/affine_type.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>

namespace tvm {

using tvm::ReprPrinter;
using namespace tvm::runtime;

TensorAffineType::TensorAffineType(RelayExpr scale, RelayExpr zero_point, DataType dtype,
                                   int axis) {
  ObjectPtr<TensorAffineTypeNode> n = make_object<TensorAffineTypeNode>();
  n->scale = std::move(scale);
  n->zero_point = std::move(zero_point);
  n->dtype = std::move(dtype);
  n->axis = std::move(axis);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TensorAffineTypeNode);

TVM_REGISTER_GLOBAL("ir.TensorAffineType")
    .set_body_typed([](RelayExpr scale, RelayExpr zero_point, DataType dtype, int axis) {
      return TensorAffineType(scale, zero_point, dtype, axis);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TensorAffineTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TensorAffineTypeNode*>(ref.get());
      p->stream << "TensorAffineType(" << node->scale << ", " << node->zero_point << ", "
                << node->dtype << ", " << node->axis << ")";
    });

TupleAffineType::TupleAffineType(Array<TensorAffineType> types) {
  ObjectPtr<TupleAffineTypeNode> n = make_object<TupleAffineTypeNode>();
  n->types = std::move(types);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TupleAffineTypeNode);

TVM_REGISTER_GLOBAL("ir.TupleAffineType").set_body_typed([](Array<TensorAffineType> types) {
  return TupleAffineType(types);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleAffineTypeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TupleAffineTypeNode*>(ref.get());
      p->stream << "TupleAffineType([";
      for (size_t i = 0; i < node->types.size(); ++i) {
        p->stream << node->types[i];
        if (i < node->types.size() - 1) {
          p->stream << ", ";
        }
      }
      p->stream << "])";
    });

}  // namespace tvm
