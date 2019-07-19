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
 *  Copyright (c) 2018 by Contributors
 * \file nn.h
 * \brief Property def of nn operators that need to be shared by quantized and unquantized ops.
 */

#ifndef TVM_NN_H
#define TVM_NN_H

#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/attrs/nn.h>
#include "../type_relations.h"
#include "../../qnn/util.h"
#include <type_traits>

namespace tvm {
namespace relay {

// relay.nn.dense
enum DenseType {
  kUnquantizedDense,
  kQuantizedDense
};

template <typename AttrType, DenseType mode>
inline bool DenseRel(const Array<Type>& types,
              int num_inputs,
              const Attrs& attrs,
              const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const auto* param = attrs.as<AttrType>();
  CHECK(param != nullptr);

  CHECK(static_cast<int>(data->shape.size()) != 0);
  if(mode == DenseType::kQuantizedDense) {
    CHECK(IsValidOpInputType(qnn::QuantizeOpType::QuantizedDense, data->dtype))
        << "Expected quantized dense type(int8, uint8) for input but was " <<  data->dtype;
    CHECK(IsValidOpInputType(qnn::QuantizeOpType::QuantizedDense, weight->dtype))
      << "Expected quantized dense type(int8, uint8) for weight but was " <<  weight->dtype;
    CHECK(data->dtype == weight->dtype) << "Weight and kernel dtypes do not match";
    CHECK(IsValidOpOutputType(qnn::QuantizeOpType::QuantizedDense, param->out_dtype))
      << "Expected quantized dense type(int32, int16) for output but was " <<  param->out_dtype;
  }
  Array<tvm::Expr> oshape = data->shape;
  if (param->units.defined()) {
    Array<tvm::Expr> dshape = data->shape;
    // validate the weight shape is proper if defined
    // Assign weight type
    Array<IndexExpr> wshape({param->units, dshape[dshape.size() - 1]});
    reporter->Assign(types[1], TensorTypeNode::make(wshape, data->dtype));
    oshape.Set((oshape.size() - 1), param->units);
  } else {
    if (weight == nullptr) return false;
    Array<tvm::Expr> wshape = weight->shape;
    oshape.Set((oshape.size() - 1), wshape[0]);
  }

  DataType out_dtype = param->out_dtype;
  if(mode == DenseType::kUnquantizedDense) {
    if (out_dtype.bits() == 0) {
      out_dtype = data->dtype;
    }
  }
  // assign output type
  reporter->Assign(types[2], TensorTypeNode::make(oshape, out_dtype));
  return true;
}

} // namespace relay
} // namespace tvm

#endif //TVM_NN_H
