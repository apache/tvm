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
 * \file sparse.cc
 * \brief Property def of nn.sparse_dense operator.
 */

#include <tvm/tir/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <vector>

#include "../../pass/infer_layout_util.h"

namespace tvm {
namespace relay {

// relay.nn.sparse_dense
TVM_REGISTER_NODE_TYPE(SparseDenseAttrs);

bool SparseDenseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight_data = types[1].as<TensorTypeNode>();
  CHECK(weight_data->shape.size() == 1 || weight_data->shape.size() == 3);
  const auto* weight_indptr = types[3].as<TensorTypeNode>();
  if (data == nullptr) return false;

  if (weight_data->shape.size() == 1) {
    // CSR case.
    Array<IndexExpr> oshape({data->shape[0], weight_indptr->shape[0] - 1});
    reporter->Assign(types[4], TensorType(oshape, data->dtype));
    return true;
  }

  if (weight_data->shape.size() == 3) {
    // BSR case.
    Array<IndexExpr> oshape({
        data->shape[0],
          (weight_indptr->shape[0] - 1) * weight_data->shape[1]});
    reporter->Assign(types[4], TensorType(oshape, data->dtype));
    return true;
  }
  LOG(FATAL) << "Unknown weight ndim for nn.sparse_dense, should be 1 (CSR) or 3 (BSR)";
  return false;
}

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeSparseDense(Expr data, Expr weight_data, Expr weight_indices, Expr weight_indptr) {
  auto attrs = make_object<SparseDenseAttrs>();
  static const Op& op = Op::Get("nn.sparse_dense");
  return CallNode::make(op, {data, weight_data, weight_indices, weight_indptr}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.sparse_dense")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
  runtime::detail::unpack_call<Expr, 4>(MakeSparseDense, args, rv);
});

RELAY_REGISTER_OP("nn.sparse_dense")
.describe(R"code(Applies a sparse linear transformation: :math:`Y = XW^T` with X sparse.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
.set_attrs_type<SparseDenseAttrs>()
.set_num_inputs(4)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight_data", "1D Tensor", "Weight data matrix.")
.add_argument("weight_indices", "1D Tensor", "Weight indices matrix.")
.add_argument("weight_indptr", "1D Tensor", "Weight indptr matrix.")
.set_support_level(1)
.add_type_rel("SparseDense", SparseDenseRel);

// relay.nn.sparse_transpose
TVM_REGISTER_NODE_TYPE(SparseTransposeAttrs);

bool SparseTransposeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);
  const auto* sparse_data = types[0].as<TensorTypeNode>();
  CHECK_EQ(sparse_data->shape.size(), 1);
  const auto* sparse_indices = types[1].as<TensorTypeNode>();
  CHECK_EQ(sparse_indices->shape.size(), 1);
  const auto* sparse_indptr = types[2].as<TensorTypeNode>();

  std::vector<Type> output_types;
  output_types.push_back(TensorType(sparse_data->shape, sparse_data->dtype));
  output_types.push_back(TensorType(sparse_indices->shape, sparse_indices->dtype));
  output_types.push_back(TensorType(sparse_indptr->shape, sparse_indptr->dtype));

  reporter->Assign(types[3], TupleType(Array<Type>(output_types)));
  return true;
}

Expr MakeSparseTranspose(Expr sparse_data, Expr sparse_indices, Expr sparse_indptr) {
  auto attrs = make_object<SparseTransposeAttrs>();
  static const Op& op = Op::Get("nn.sparse_transpose");
  return CallNode::make(op, {sparse_data, sparse_indices, sparse_indptr}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.sparse_transpose")
.set_body_typed(MakeSparseTranspose);


RELAY_REGISTER_OP("nn.sparse_transpose")
.describe(R"code(Transpose a sparse matrix X. Only support square sparse matrix

- **input**: `(N, N)`
- **out**: `(N, N)`.

)code" TVM_ADD_FILELINE)
.set_attrs_type<SparseTransposeAttrs>()
.set_num_inputs(3)
.add_argument("sparse_data", "1D Tensor", "Sparse data matrix.")
.add_argument("sparse_indices", "1D Tensor", "Sparse indices matrix.")
.add_argument("sparse_indptr", "1D Tensor", "Sparse index pointer matrix.")
.set_support_level(1)
.add_type_rel("SparseTranspose", SparseTransposeRel);

}  // namespace relay
}  // namespace tvm
