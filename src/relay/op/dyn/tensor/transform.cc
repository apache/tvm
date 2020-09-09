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

#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/data_layout.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/transform.h>

#include <string>
#include <utility>
#include <vector>

#include "../../../transforms/infer_layout_util.h"

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

// broadcast_to operator
bool BroadCastToRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  // types = [data_type, broadcast_shape_type, ret_type]
  CHECK_EQ(types.size(), 3);

  const auto* target_shape = types[1].as<TensorTypeNode>();
  DataType out_dtype = types[0].as<TensorTypeNode>()->dtype;
  // rank must be static
  const IntImmNode* rank = target_shape->shape[0].as<IntImmNode>();
  CHECK(rank) << "Target shape must have static rank";  // rank must be static even in dyn pass
                                                        // could add support for dyn rank in futures

  std::vector<IndexExpr> oshape;
  for (int i = 0; i < rank->value; ++i) {
    oshape.push_back(Any());
  }

  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

Expr MakeBroadCastTo(Expr data, Expr shape) {
  static const Op& op = Op::Get("dyn.broadcast_to");
  auto attrs = make_object<InitOpAttrs>();
  return Call(op, {data, shape}, Attrs(attrs), {});
}

Array<te::Tensor> BroadCastToCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                     const Type& out_type) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  return {topi::broadcast_to(inputs[0], out_ttype->shape)};
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.broadcast_to").set_body_typed(MakeBroadCastTo);

RELAY_REGISTER_OP("dyn.broadcast_to")
    .describe(R"code(Broadcast the first input to match the shape argument.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Tensor", "Target shape.")
    .set_support_level(4)
    .add_type_rel("DynamicBroadCastTo", BroadCastToRel)
    .set_attr<FTVMCompute>("FTVMCompute", BroadCastToCompute)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

// zeros and ones operator
bool InitOpRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types = [zeros_shape, ret_type]
  CHECK_EQ(types.size(), 2);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  const auto* fill_shape = types[0].as<TensorTypeNode>();
  DataType out_dtype = param->dtype;

  const IntImmNode* shape_shape = fill_shape->shape[0].as<IntImmNode>();
  CHECK(shape_shape) << "Parameter shape must have static rank";

  std::vector<IndexExpr> oshape;
  for (int i = 0; i < shape_shape->value; ++i) {
    oshape.push_back(Any());
  }

  reporter->Assign(types[1], TensorType(oshape, out_dtype));
  return true;
}

Expr MakeZeros(Expr shape, DataType dtype) {
  auto attrs = make_object<InitOpAttrs>();
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("dyn.zeros");
  return Call(op, {shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.zeros").set_body_typed(MakeZeros);

RELAY_REGISTER_OP("dyn.zeros")
    .describe(R"code(Fill array with zeros.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<InitOpAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Tensor", "Target shape.")
    .set_support_level(3)
    .add_type_rel("DynamicInitOp", InitOpRel);

Expr MakeOnes(Expr shape, DataType dtype) {
  auto attrs = make_object<InitOpAttrs>();
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("dyn.ones");
  return Call(op, {shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.ones").set_body_typed(MakeOnes);

RELAY_REGISTER_OP("dyn.ones")
    .describe(R"code(Fill array with ones.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<InitOpAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Tensor", "Target shape.")
    .set_support_level(3)
    .add_type_rel("DynamicInitOp", InitOpRel);

bool OneHotRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // `types` contains: [indices, on_value, off_value, result]
  CHECK_EQ(types.size(), 5);
  const auto* indices = types[0].as<TensorTypeNode>();
  CHECK(indices);

  const auto param = attrs.as<OneHotAttrs>();

  Array<IndexExpr> oshape;
  int ndim = indices->shape.size() + 1;
  int indices_index = 0;
  int true_axis = (param->axis == -1) ? indices->shape.size() : param->axis;
  for (int i = 0; i < ndim; i++) {
    if (i == true_axis) {
      oshape.push_back(Any());
    } else {
      oshape.push_back(indices->shape[indices_index++]);
    }
  }

  reporter->Assign(types[4], TensorType(oshape, param->dtype));
  return true;
}

Array<te::Tensor> OneHotCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  const auto* param = attrs.as<OneHotAttrs>();
  CHECK(param != nullptr);
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  return Array<te::Tensor>{topi::one_hot(inputs[0], inputs[1](), inputs[2](), -1, param->axis,
                                         param->dtype, out_ttype->shape)};
}

Expr MakeOneHot(Expr indices, Expr on_value, Expr off_value, Expr depth, int axis, DataType dtype) {
  auto attrs = make_object<OneHotAttrs>();
  attrs->axis = axis;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("dyn.one_hot");
  return Call(op, {indices, on_value, off_value, depth}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.one_hot").set_body_typed(MakeOneHot);

RELAY_REGISTER_OP("dyn.one_hot")
    .describe(R"code(Returns a one-hot tensor where the locations repsented by indices take value 1,
    other locations take value 0. Final dimension is <indices dimensions> x depth.

    **indices** Locations to set to 1.

    **on_value** Value to fill at indices.

    **off_value** Value to fill at all other positions besides indices.

    **depth** Depth of the one-hot dimension.

    **axis** Axis to fill.

    **dtype**)code" TVM_ADD_FILELINE)
    .set_attrs_type<OneHotAttrs>()
    .set_num_inputs(4)
    .add_argument("indices", "Tensor", "Locations to set to on_value.")
    .add_argument("on_value", "Expr", "Value to fill at indices.")
    .add_argument("off_value", "Expr", "Value to fill at all other positions besides indices.")
    .add_argument("depth", "Expr", "Value to fill at all other positions besides indices.")
    .set_support_level(10)
    .add_type_rel("DynOneHot", OneHotRel)
    .set_attr<FTVMCompute>("FTVMCompute", OneHotCompute)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

bool FullRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  const auto* fill_value = types[0].as<TensorTypeNode>();
  const auto* fill_shape = types[1].as<TensorTypeNode>();
  if (fill_value == nullptr) {
    return false;
  }

  DataType out_dtype = param->dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = fill_value->dtype;
  }

  CHECK_EQ(fill_value->shape.size(), 0)
      << "Fill value should be a scalar but has dimension " << fill_value->shape.size() << ".";

  const IntImmNode* rank = fill_shape->shape[0].as<IntImmNode>();
  CHECK(rank) << "Parameter shape must have static rank";

  std::vector<IndexExpr> oshape;
  for (int i = 0; i < rank->value; ++i) {
    oshape.push_back(Any());
  }
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

Expr MakeFull(Expr fill_value, Expr shape, DataType dtype) {
  auto attrs = make_object<InitOpAttrs>();
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("dyn.full");
  return Call(op, {fill_value, shape}, Attrs(attrs), {});
}
Array<te::Tensor> FullCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  return {topi::full(out_ttype->shape, out_ttype->dtype, inputs[0]())};
}
TVM_REGISTER_GLOBAL("relay.op.dyn._make.full").set_body_typed(MakeFull);

RELAY_REGISTER_OP("dyn.full")
    .describe(R"code(Fill array with scalar value.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<InitOpAttrs>()
    .set_num_inputs(2)
    .add_argument("fill_value", "double", "The value to fill.")
    .add_argument("shape", "Tensor", "Target shape.")
    .set_support_level(3)
    .add_type_rel("DynamicFull", FullRel)
    .set_attr<FTVMCompute>("FTVMCompute", FullCompute)
    .set_attr<TOpPattern>("TOpPattern", kElemWise);

bool StridedSliceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  // [data, begin, end, strides, out]
  CHECK_EQ(types.size(), 5);
  const StridedSliceAttrs* param = attrs.as<StridedSliceAttrs>();
  if (param == nullptr) {
    return false;
  }
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  auto dshape = data->shape;
  int64_t num_axis = dshape.size();

  // calculate output shape
  std::vector<IndexExpr> oshape(num_axis);
  for (int64_t i = 0; i < num_axis; ++i) {
    oshape[i] = Any();
  }

  reporter->Assign(types[4], TensorType(oshape, data->dtype));
  return true;
}

inline te::Tensor DynamicStridedSlice(const te::Tensor& input, const te::Tensor& begin,
                                      const te::Tensor& end, const te::Tensor& strides,
                                      std::string name = "T_strided_slice_dynamic",
                                      std::string tag = topi::kInjective) {
  int64_t src_tensor_dim = input->shape.size();
  Array<IndexExpr> out_shape;
  for (int64_t i = 0; i < src_tensor_dim; ++i) {
    out_shape.push_back(tvm::tir::Var("dim"));
  }
  // TODO(yongwww): move the compute into topi
  return te::compute(
      out_shape,
      [&](const Array<tvm::tir::Var>& indices) {
        Array<IndexExpr> real_indices;
        for (int32_t i = 0; i < src_tensor_dim; ++i) {
          real_indices.push_back(indices[i] * strides(i) + begin(i));
        }
        return input(real_indices);
      },
      name, tag);
}

Array<te::Tensor> StridedSliceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                      const Type& out_type) {
  te::Tensor data = inputs[0];
  te::Tensor begin = inputs[1];
  te::Tensor end = inputs[2];
  te::Tensor strides = inputs[3];
  // Dynamic computation
  int64_t data_rank = data->shape.size();
  CHECK(begin->shape[0].as<IntImmNode>()->value == data_rank &&
        end->shape[0].as<IntImmNode>()->value == data_rank &&
        strides->shape[0].as<IntImmNode>()->value == data_rank)
      << "begin, end, and strides are required to have the same length"
      << " if they are dynamic variables.";
  return Array<te::Tensor>{DynamicStridedSlice(data, begin, end, strides)};
}

Expr MakeStridedSlice(Expr data, Expr begin, Expr end, Expr strides, String slice_mode) {
  auto attrs = make_object<StridedSliceAttrs>();
  attrs->slice_mode = slice_mode;
  static const Op& op = Op::Get("dyn.strided_slice");
  return Call(op, {data, begin, end, strides}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.strided_slice").set_body_typed(MakeStridedSlice);

RELAY_REGISTER_OP("dyn.strided_slice")
    .describe(R"code(Strided slice of an array.

Examples::

  x = [[  1.,   4.,   7.,  10.],
       [  2.,   5.,   8.,  11.],
       [  3.,   6.,   9.,  12.]]

  strided_slice(x, begin=[0, 1], end=[2, 4], stride=[1, 1]) = [[ 4.,  7.,  10.],
                                                               [ 5.,  8.,  11.]]

  x = [[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]]]

  strided_slice(x, begin=[0, 0], end=[2, 2]) = [[[ 1.,  2.],
                                                 [ 3.,  4.]],

                                                [[ 5.,  6.],
                                                 [ 7.,  8.]]]
)code" TVM_ADD_FILELINE)
    .set_num_inputs(4)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("begin", "Tensor", "The indices to begin with in the slicing.")
    .add_argument("end", "Tensor", "Indices indicating end of the slice.")
    .add_argument("strides", "Tensor", "The stride values.")
    .add_argument("slice_mode", "Tensor", "The slice mode.")
    .set_support_level(4)
    .set_attrs_type<StridedSliceAttrs>()
    .add_type_rel("DynStridedSlice", StridedSliceRel)
    .set_attr<FTVMCompute>("FTVMCompute", StridedSliceCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_attr<AnyCodegenStrategy>("AnyCodegenStrategy", kVariableDimensions);

}  // namespace dyn
}  // namespace relay
}  // namespace tvm
