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
 * \file src/relay/op/nn/nn.h
 * \brief Properties def of nn operators for sharing.
 */
#ifndef TVM_RELAY_OP_NN_NN_H_
#define TVM_RELAY_OP_NN_NN_H_

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/relay/type.h>

#include <algorithm>
#include <utility>

#include "../op_common.h"

namespace tvm {
namespace relay {

template <typename AttrType>
bool MatmulRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* tensor_a = types[0].as<TensorTypeNode>();
  const auto* tensor_b = types[1].as<TensorTypeNode>();
  if (tensor_a == nullptr) return false;
  ICHECK(static_cast<int>(tensor_a->shape.size()) != 0);

  const AttrType* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);
  // Default set to dense layout
  bool transpose_a = false;
  bool transpose_b = true;
  const auto& mattrs = attrs.as<MatmulAttrs>();
  if (mattrs != nullptr) {
    transpose_a = mattrs->transpose_a;
    transpose_b = mattrs->transpose_b;
  }

  const Array<tvm::PrimExpr>& dshape = tensor_a->shape;
  Array<tvm::PrimExpr> oshape = dshape;
  tvm::PrimExpr reduce = dshape[dshape.size() - 1];
  if (transpose_a) {
    reduce = dshape[dshape.size() - 2];
    oshape.Set((oshape.size() - 2), dshape[oshape.size() - 1]);
  }
  if (param->units.defined()) {
    // validate the tensor_b shape is proper if defined
    // Assign tensor_b type
    const Array<IndexExpr>& wshape = transpose_b ? Array<IndexExpr>({param->units, reduce})
                                                 : Array<IndexExpr>({reduce, param->units});
    // It is possible for tensor_b to be nullptr in which case we will use
    // data dtype as the tensor_b dtype. However if tensor_b dtype is explicitly
    // present we will use that.
    auto tensor_b_dtype = (tensor_b == nullptr ? tensor_a->dtype : tensor_b->dtype);
    if (param->auto_scheduler_rewritten_layout.size() == 0) {
      // Normal case: assign result to reporter
      reporter->Assign(types[1], TensorType(wshape, tensor_b_dtype));
    } else {
      // If the layout is rewritten by auto-scheduler,
      // we just forcly apply the layout provided by auto-scheduler and
      // skip the normal inference logic.
      {}  // do nothing
    }
    oshape.Set((oshape.size() - 1), param->units);
  } else {
    if (tensor_b == nullptr) return false;
    const Array<tvm::PrimExpr>& wshape = tensor_b->shape;
    // When tensor_b's layout has been rewritten, figure it out based on the
    // total number of elements and input dimensions.
    if (param->auto_scheduler_rewritten_layout.size() != 0) {
      PrimExpr tensor_b_elements = 1;
      for (size_t i = 0; i < wshape.size(); i++) {
        tensor_b_elements = tensor_b_elements * wshape[i];
      }
      oshape.Set(oshape.size() - 1, tensor_b_elements / dshape[dshape.size() - 1]);
      // Otherwise just pull it out of the tensor_b shape directly.
    } else {
      ICHECK(static_cast<int>(tensor_b->shape.size()) == 2);
      if (!tensor_a->shape.back().as<tir::AnyNode>()) {
        ICHECK((transpose_b && reporter->AssertEQ(reduce, tensor_b->shape[1])) ||
               (!transpose_b && reporter->AssertEQ(reduce, tensor_b->shape[0])))
            << "MatmulRel: input dimension doesn't match,"
            << " tensor_a shape=" << tensor_a->shape << ", tensor_b shape=" << tensor_b->shape;
      }
      oshape.Set((oshape.size() - 1), transpose_b ? wshape[0] : wshape[1]);
    }
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = tensor_a->dtype;
  }
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

template <typename AttrType>
bool DensePackRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr || weight == nullptr) return false;

  const AttrType* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);

  Array<tvm::PrimExpr> oshape = data->shape;
  oshape.Set((oshape.size() - 1), weight->shape[0] * weight->shape[2]);

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

template <typename AttrType>
bool BatchMatmulRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* x = types[0].as<TensorTypeNode>();
  const auto* y = types[1].as<TensorTypeNode>();
  if (x == nullptr || y == nullptr) return false;

  const AttrType* param = attrs.as<AttrType>();
  Array<PrimExpr> y_shape;
  if (param->auto_scheduler_rewritten_layout.size() == 0) {
    y_shape = y->shape;
  } else {
    y_shape = auto_scheduler::GetShapeFromRewrittenLayout(param->auto_scheduler_rewritten_layout,
                                                          {"b", "j", "k"});
  }

  ICHECK(x->shape.size() == 3 && y_shape.size() == 3);
  bool is_dyn = false;
  Array<tvm::PrimExpr> oshape;
  for (size_t i = 0; i < 3; ++i) {
    if (x->shape[i].as<tir::AnyNode>() != nullptr || y_shape[i].as<tir::AnyNode>() != nullptr) {
      is_dyn = true;
      oshape.push_back(Any());
    } else {
      if (i == 0) {
        oshape.push_back(max(x->shape[i], y_shape[i]));
      } else {
        oshape.push_back(x->shape[i]);
      }
    }
  }
  if (!is_dyn) {
    ICHECK(reporter->AssertEQ(x->shape[0], y_shape[0]) || reporter->AssertEQ(x->shape[0], 1) ||
           reporter->AssertEQ(y_shape[0], 1))
        << "BatchDot: batch dimensions don't match, "
        << " x shape=" << x->shape << ", y shape=" << y_shape;
    ICHECK(reporter->AssertEQ(x->shape[2], y_shape[2]))
        << "BatchDot: shapes of x and y is inconsistent, "
        << " x shape=" << x->shape << ", y shape=" << y_shape;
  }
  oshape.Set(2, y_shape[1]);

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = x->dtype;
  }
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_NN_NN_H_
