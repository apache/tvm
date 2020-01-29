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
 * \file src/relay/op/tensor/transform.h
 * \brief Transform op attributes that can be shared among Relay and its dialects.
 */
#ifndef TVM_RELAY_OP_TENSOR_TRANSFORM_H_
#define TVM_RELAY_OP_TENSOR_TRANSFORM_H_

#include <tvm/ir/error.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace relay {

template <typename AttrType>
bool ConcatenateRel(const Array<Type>& types,
                    int num_inputs,
                    const Attrs& attrs,
                    const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  /* If we receive a tuple we can continue, if we receive
   * anything but an incomplete type we should signal an
   * error.
  */
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    throw Error(
        ErrorBuilder()
        << "concatenate requires a tuple of tensors as the first argument, found "
        << PrettyPrint(types[0]));
  } else if (types[0].as<IncompleteTypeNode>() != nullptr) {
    return false;
  }

  const auto* param = attrs.as<AttrType>();
  if (tensor_tuple->fields[0].as<IncompleteTypeNode>()) {
    return false;
  }
  const auto& first = Downcast<TensorType>(tensor_tuple->fields[0]);
  // Sanity check: ndim and dtype.
  const int ndim = static_cast<int>(first->shape.size());
  const DataType dtype = first->dtype;

  // Sanity check: axis
  int axis = param->axis;
  if (!(-ndim <= axis && axis < ndim)) {
    throw Error(ErrorBuilder() <<
      "concatenate only accepts `axis` in [-ndim, ndim)" <<
      ", but got axis = " << axis <<
      ", and ndim = " << ndim);
  }
  axis = axis < 0 ? ndim + axis : axis;

  for (const Type& ele : tensor_tuple->fields) {
    if (ele.as<IncompleteTypeNode>()) {
      return false;
    }

    const auto& e = Downcast<TensorType>(ele);

    int e_ndim = static_cast<int>(e->shape.size());
    const DataType& e_dtype = e->dtype;
    if (e_ndim != ndim) {
      throw Error("relay.concatenate requires all tensors have the same ndim");
    }
    if (e_dtype != dtype) {
      throw Error("relay.concatenate requires all tensors have the same dtype");
    }
    for (size_t j = 0; j < first->shape.size(); ++j) {
      if (j == static_cast<size_t>(axis)) continue;
      if (reporter->AssertEQ(first->shape[j], e->shape[j])) continue;
      throw Error("relay.concatenate requires all tensors have the same shape "
                   "on non-concatenating axes");
    }
  }

  // Calculate shape
  std::vector<IndexExpr> oshape(first->shape.begin(), first->shape.end());
  IndexExpr &concat_dim = oshape[axis];
  bool has_any = false;
  if (concat_dim.as<Any>()) {
    has_any = true;
  } else {
    for (int i = 1; i < static_cast<int>(tensor_tuple->fields.size()); ++i) {
      const auto& e = Downcast<TensorType>(tensor_tuple->fields[i]);
      if (e->shape[axis].as<Any>()) {
        has_any = true;
        break;
      }
      concat_dim += e->shape[axis];
    }
  }

  if (has_any) {
    concat_dim = Any::make();
  }

  auto rtype = TensorType(oshape, dtype);
  reporter->Assign(types[1], rtype);
  return true;
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_TENSOR_TRANSFORM_H_
