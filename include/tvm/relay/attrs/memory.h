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
 * \file tvm/relay/attrs/memory.h
 * \brief Attributes for memory operators.
 */
#ifndef TVM_RELAY_ATTRS_MEMORY_H_
#define TVM_RELAY_ATTRS_MEMORY_H_

#include <tvm/ir/attrs.h>
#include <tvm/relay/expr.h>
#include <string>
#include <vector>

namespace tvm {
namespace relay {

std::vector<TensorType> FlattenTupleType(const Type& type);
std::vector<Expr> FromTupleType(const Type& type, const Expr& expr);
Expr ToTupleType(const Type& t, const Array<Expr>& exprs);

/*!
 * \brief Options for allocating storage.
 */
struct AllocStorageAttrs : public tvm::AttrsNode<AllocStorageAttrs> {
  DataType dtype;
  int device_id;
  int device_type;

  TVM_DECLARE_ATTRS(AllocStorageAttrs, "relay.attrs.AllocStorageAttrs") {
    TVM_ATTR_FIELD(dtype)
      .describe(
         "The dtype of the tensor to allocate.")
      .set_default(DataType::Float(32, 1));
    TVM_ATTR_FIELD(device_id)
      .describe(
        "The device id on which to allocate memory.");
    TVM_ATTR_FIELD(device_type)
      .describe(
        "The device type on which to allocate memory.");
  }
};

/*!
 * \brief Options for allocating tensors.
 */
struct AllocTensorAttrs : public tvm::AttrsNode<AllocTensorAttrs> {
  Constant const_shape;
  Array<IndexExpr> assert_shape;
  DataType dtype;

  TVM_DECLARE_ATTRS(AllocTensorAttrs, "relay.attrs.AllocTensorAttrs") {
    TVM_ATTR_FIELD(dtype)
      .describe(
         "The dtype of the tensor to allocate.")
      .set_default(DataType::Float(32, 1));
    TVM_ATTR_FIELD(const_shape)
      .describe(
         "The shape of constant used to aid in type inference.");
    TVM_ATTR_FIELD(assert_shape)
      .describe(
         "The shape to cast the return type of the allocation to, "\
         "used to specify the shape obtained via further analysis.");
  }
};

/*!
 * \brief Options for the shape function operator.
 */
struct ShapeFuncAttrs : public tvm::AttrsNode<ShapeFuncAttrs> {
  Array<Integer> is_input;

  TVM_DECLARE_ATTRS(ShapeFuncAttrs, "relay.attrs.ShapeFuncAttrs") {
    TVM_ATTR_FIELD(is_input)
      .describe(
         "A bool indicating whether the shape function should"\
         "expect shape or input in each position.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_MEMORY_H_
