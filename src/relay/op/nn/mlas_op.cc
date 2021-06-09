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
 * \file mlas_op.cc
 * \brief Implementation of operators from MLAS library
 */

#include <tvm/relay/attrs/mlas_op.h>

#include "../op_common.h"

namespace tvm {
namespace relay {

// relay.mlas_matmul
TVM_REGISTER_NODE_TYPE(MlasMatmulAttrs);

bool MlasMatmulRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* A = types[0].as<TensorTypeNode>();
  const auto* B = types[1].as<TensorTypeNode>();
  if (A == nullptr || B == nullptr) return false;
  const auto* param = attrs.as<MlasMatmulAttrs>();

  Array<tvm::PrimExpr> oshape;
  // If B matrix is pre-packed then it is 1-D
  if (!param->packb) {
    ICHECK_EQ(A->shape.size(), B->shape.size());
  }
  bool is_dyn = false;
  if (!param->packb) {           // When B is not pre-packed
    if (A->shape.size() == 3) {  // The case of batch_matmul A[:,:,:] x B[:,:,:]^T
      // batch
      if (A->shape[0].as<tir::AnyNode>() != nullptr || B->shape[0].as<tir::AnyNode>() != nullptr) {
        is_dyn = true;
        oshape.push_back(Any());
      } else {
        oshape.push_back(max(A->shape[0], B->shape[0]));
      }
      // M
      if (A->shape[1].as<tir::AnyNode>() != nullptr) {
        is_dyn = true;
        oshape.push_back(Any());
      } else {
        oshape.push_back(A->shape[1]);
      }
      // N
      if (B->shape[1].as<tir::AnyNode>() != nullptr) {
        is_dyn = true;
        oshape.push_back(Any());
      } else {
        oshape.push_back(B->shape[1]);
      }

      if (!is_dyn) {
        ICHECK(reporter->AssertEQ(A->shape[0], B->shape[0]) || reporter->AssertEQ(A->shape[0], 1) ||
               reporter->AssertEQ(B->shape[0], 1))
            << "MlasMatmulRel: batch dimensions don't match, "
            << " A shape=" << A->shape << ", B shape=" << B->shape;
        ICHECK(reporter->AssertEQ(A->shape[2], B->shape[2]))
            << "MlasMatmulRel: shapes of A and B are inconsistent, "
            << " A shape=" << A->shape << ", B shape=" << B->shape;
      }
    } else {  // The case of dense A[:,:] x B[:,:]^T
      // M
      if (A->shape[0].as<tir::AnyNode>() != nullptr) {
        is_dyn = true;
        oshape.push_back(Any());
      } else {
        oshape.push_back(A->shape[0]);
      }
      // N
      if (B->shape[0].as<tir::AnyNode>() != nullptr) {
        is_dyn = true;
        oshape.push_back(Any());
      } else {
        oshape.push_back(B->shape[0]);
      }
      if (!is_dyn) {
        ICHECK(reporter->AssertEQ(A->shape[1], B->shape[1]))
            << "MlasMatmulRel: shapes of A and B are inconsistent, "
            << " A shape=" << A->shape << ", B shape=" << B->shape;
      }
    }
  } else {                       // When B is pre-packed, B is 1-D and the batch_size of B must be 1
    if (A->shape.size() == 3) {  // The case of batch_matmul A[:,:,:] x B[:,:,:]^T
      // batch
      if (A->shape[0].as<tir::AnyNode>() != nullptr) {
        is_dyn = true;
        oshape.push_back(Any());
      } else {
        oshape.push_back(A->shape[0]);
      }
      // M
      if (A->shape[1].as<tir::AnyNode>() != nullptr) {
        is_dyn = true;
        oshape.push_back(Any());
      } else {
        oshape.push_back(A->shape[1]);
      }
      // N
      oshape.push_back(param->N);
      ICHECK(reporter->AssertEQ(A->shape[2], param->K))
          << "MlasMatmulRel: shapes of A and B are inconsistent, "
          << " A shape=" << A->shape << ", B shape="
          << "[1," << param->N << "," << param->K << "]";
    } else {  // The case of dense A[:,:] x B[:,:]^T
      // M
      if (A->shape[0].as<tir::AnyNode>() != nullptr) {
        is_dyn = true;
        oshape.push_back(Any());
      } else {
        oshape.push_back(A->shape[0]);
      }
      // N
      oshape.push_back(param->N);
      ICHECK(reporter->AssertEQ(A->shape[1], param->K))
          << "MlasMatmulRel: shapes of A and B are inconsistent, "
          << " A shape=" << A->shape << ", B shape="
          << "[" << param->N << "," << param->K << "]";
    }
  }
  reporter->Assign(types[2], TensorType(oshape, A->dtype));
  return true;
}

Expr MakeMlasMatmul(Expr x, Expr y, bool packb, int K, int N) {
  auto attrs = make_object<MlasMatmulAttrs>();
  attrs->packb = packb;
  attrs->K = K;
  attrs->N = N;
  static const Op& op = Op::Get("mlas_matmul");
  return Call(op, {x, y}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.mlas_matmul").set_body_typed(MakeMlasMatmul);

RELAY_REGISTER_OP("mlas_matmul")
    .describe(R"code(Computes matrix multiplication using mlas library

.. math::

  batch\_matmul(A, B)[i, :, :] = matmul(A[i, :, :], B[i, :, :]^T)
  or batch\_matmul(A, B)[:, :] = matmul(A[:, :], B[:, :]^T)

- **A**: `(b, m, k)` or `(m, k)`
- **B**: `(b, n, k)` or `(n, k)`
- **out**: `(b, m, n)` or `(m, n)`.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("A", "3D/2D Tensor", "First input.")
    .add_argument("B", "3D/2D Tensor", "Second input.")
    .set_support_level(10)
    .add_type_rel("MlasMatmul", MlasMatmulRel);

// relay.mlas_packb
TVM_REGISTER_NODE_TYPE(MlasPackbAttrs);

bool MlasPackbRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* B = types[0].as<TensorTypeNode>();
  if (B == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "MlasPackbRel: expect input data type to be TensorType but get " << types[0];
    return false;
  }
  const MlasPackbAttrs* params = attrs.as<MlasPackbAttrs>();
  reporter->Assign(types[1], TensorType({params->size}, B->dtype));
  return true;
}

Expr MakeMlasPackb(Expr B, int K, int N, int size, bool transb) {
  auto attrs = make_object<MlasPackbAttrs>();
  attrs->K = K;
  attrs->N = N;
  attrs->size = size;
  attrs->transb = transb;
  static const Op& op = Op::Get("mlas_packb");
  return Call(op, {B}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.mlas_packb").set_body_typed(MakeMlasPackb);

RELAY_REGISTER_OP("mlas_packb")
    .describe(R"code(Pre-pack the B matrix
)code" TVM_ADD_FILELINE)
    .set_attrs_type<MlasPackbAttrs>()
    .set_num_inputs(1)
    .add_argument("B", "Tensor", "The second matrix of matmul.")
    .add_type_rel("mlas_packb", MlasPackbRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace relay
}  // namespace tvm
