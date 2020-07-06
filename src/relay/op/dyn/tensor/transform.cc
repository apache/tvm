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
 * \file transform.cc
 * \brief Dynamic Transform operators.
 */
#include "transform.h"

#include <topi/transform.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/registry.h>

#include <vector>

namespace tvm {
namespace relay {
namespace dyn {

/* relay.dyn.reshape */
bool ReshapeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  // types: [data, newshape, result]
  CHECK_EQ(types.size(), 3);

  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "reshape: expect input type to be TensorType but get " << types[0];
    return false;
  }

  Array<IndexExpr> oshape;
  const auto* newshape = types[1].as<TensorTypeNode>();

  // Doesn't support dynamic output rank
  for (int i = 0; i < newshape->shape[0].as<IntImmNode>()->value; i++) {
    oshape.push_back(Any());
  }

  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> ReshapeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  CHECK(out_ttype != nullptr);
  Array<IndexExpr> newshape;
  for (auto val : out_ttype->shape) {
    if (val->IsInstance<tir::AnyNode>()) {
      newshape.push_back(val.as<tir::AnyNode>()->ToVar());
    } else {
      newshape.push_back(val);
    }
  }
  return {topi::reshape(inputs[0], newshape)};
}

Expr MakeReshape(Expr data, Expr newshape) {
  auto attrs = make_object<ReshapeAttrs>();
  attrs->reverse = false;
  static const Op& op = Op::Get("dyn.reshape");
  return Call(op, {data, newshape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.reshape").set_body_typed(MakeReshape);

RELAY_REGISTER_OP("dyn.reshape")
    .describe(R"code(Reshapes the input array based on the values in the newshape array.
    
    To give user more convenience in without doing manual shape inference,
    some dimensions of the shape can take special values from the set {0, -1, -3}.
    The significance of each is explained below:

    ``0`` copy this dimension from the input to the output shape.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (4,0,2), result.shape = (4,3,2)
            data.shape = (2,3,4), newshape = (2,0,0), result.shape = (2,3,4)

    ``-1`` infers the dimension of the output shape by using the remainder of
    the input dimensions keeping the size of the new array same as that of the input array.
    At most one dimension of shape can be -1.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (6,1,-1), result.shape = (6,1,4)
            data.shape = (2,3,4), newshape = (3,-1,8), result.shape = (3,1,8)
            data.shape = (2,3,4), newshape = (-1,), result.shape = (24,)

    ``-3`` use the product of two consecutive dimensions of the input shape
    as the output dimension.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (-3,4), result.shape = (6,4)
            data.shape = (2,3,4,5), newshape = (-3,-3), result.shape = (6,20)
            data.shape = (2,3,4), newshape = (0,-3), result.shape = (2,12)

    Special values -2 and -4 from the standard reshape op would introduce dynamic rank 
    in this op. Thus, they are not permitted.

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attrs_type<ReshapeAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("newshape", "Tensor", "The shape of output tensor.")
    .set_support_level(3)
    .add_type_rel("DynamicReshape", ReshapeRel)
    .set_attr<FTVMCompute>("FTVMCompute", ReshapeCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// tile operator
// TVM_REGISTER_NODE_TYPE(TileAttrs);

bool TileRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  // `types` contains: [data, reps, result]
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* reps = types[1].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "tile: expect input type to be TensorType but get " << types[0];
    return false;
  }
  if (reps == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
        << "tile: expect input type to be TensorType but get " << types[1];
    return false;
  }
  const IntImmNode* reps_shape = reps->shape[0].as<IntImmNode>();
  CHECK(reps_shape) << "Parameter reps must have static shape";
  const size_t ndim = data->shape.size();
  const size_t rndim = reps_shape->value;
  size_t tndim = (ndim > rndim) ? ndim : rndim;
  std::vector<IndexExpr> oshape;
  oshape.reserve(tndim);
  for (size_t i = 0; i < tndim; ++i) {
    oshape.emplace_back(Any());
  }
  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> TileCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  CHECK_EQ(inputs.size(), 2);
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  size_t rndim = inputs[1]->shape[0].as<IntImmNode>()->value;
  return {topi::dyn_tile(inputs[0], out_ttype->shape, rndim)};
}

Expr MakeTile(Expr data, Expr reps) {
  auto attrs = make_object<TileAttrs>();
  static const Op& op = Op::Get("dyn.tile");
  return Call(op, {data, reps}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.tile").set_body_typed(MakeTile);

RELAY_REGISTER_OP("dyn.tile")
    .describe(R"code(Repeat the whole array multiple times.

- **data**: The input data to the operator.
- **reps**: The number of times to repeat the operator.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attrs_type<TileAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("reps", "Tensor", "The number of times to repeat the input on each axis.")
    .set_support_level(3)
    .add_type_rel("DynamicTile", TileRel)
    .set_attr<FTVMCompute>("FTVMCompute", TileCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace dyn
}  // namespace relay
}  // namespace tvm
