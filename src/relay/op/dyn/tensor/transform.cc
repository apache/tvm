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

#include "../../../transforms/infer_layout_utils.h"

namespace tvm {
namespace relay {
namespace dyn {

/* relay.dyn.reshape */

bool ReshapeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  // types: [data, newshape, result]
  ICHECK_EQ(types.size(), 3);

  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "reshape: expect input type to be TensorType but get " << types[0];
    return false;
  }

  Array<IndexExpr> oshape;
  const auto* newshape = types[1].as<TensorTypeNode>();
  if (newshape == nullptr) {
    ICHECK(types[1].as<IncompleteTypeNode>())
        << "reshape: expect input type to be TensorType but get " << types[1];
    return false;
  }

  const IntImmNode* rank = newshape->shape[0].as<IntImmNode>();
  ICHECK(rank != nullptr) << "Dynamic Reshape doesn't support Dynamic Rank";
  for (int i = 0; i < rank->value; i++) {
    oshape.push_back(Any());
  }

  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> ReshapeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  ICHECK(out_ttype != nullptr);
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

Expr MakeReshape(Expr data, Expr newshape, bool allowzero = false) {
  auto attrs = make_object<ReshapeAttrs>();
  attrs->allowzero = allowzero;
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
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_attr<TReshapeOp>("TReshapeOp", true);

// tile operator
// TVM_REGISTER_NODE_TYPE(TileAttrs);

bool TileRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  // `types` contains: [data, reps, result]
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* reps = types[1].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "tile: expect input type to be TensorType but get " << types[0];
    return false;
  }
  if (reps == nullptr) {
    ICHECK(types[1].as<IncompleteTypeNode>())
        << "tile: expect input type to be TensorType but get " << types[1];
    return false;
  }
  const IntImmNode* reps_shape = reps->shape[0].as<IntImmNode>();
  ICHECK(reps_shape) << "Parameter reps must have static shape";
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
  ICHECK_EQ(inputs.size(), 2);
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
  ICHECK_EQ(types.size(), 3);

  const auto* input_type = types[0].as<TensorTypeNode>();
  const auto* target_type = types[1].as<TensorTypeNode>();
  if (target_type == nullptr) {
    return false;
  }
  if (input_type == nullptr) {
    return false;
  }
  auto out_dtype = input_type->dtype;
  // rank must be static
  const IntImmNode* rank = target_type->shape[0].as<IntImmNode>();
  ICHECK(rank)
      << "Target shape must have static rank";  // rank must be static even in dyn pass
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
    .set_attrs_type<InitOpAttrs>()
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
  ICHECK_EQ(types.size(), 2);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  const auto* fill_shape = types[0].as<TensorTypeNode>();
  DataType out_dtype = param->dtype;

  const IntImmNode* shape_shape = fill_shape->shape[0].as<IntImmNode>();
  ICHECK(shape_shape) << "Parameter shape must have static rank";

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
  ICHECK_EQ(types.size(), 5);
  const auto* indices = types[0].as<TensorTypeNode>();
  ICHECK(indices);

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
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 3);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  const auto* fill_value = types[0].as<TensorTypeNode>();
  const auto* fill_shape = types[1].as<TensorTypeNode>();
  if (fill_value == nullptr) {
    return false;
  }
  if (fill_shape == nullptr) {
    return false;
  }

  DataType out_dtype = param->dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = fill_value->dtype;
  }

  ICHECK_EQ(fill_value->shape.size(), 0)
      << "Fill value should be a scalar but has dimension " << fill_value->shape.size() << ".";

  const IntImmNode* rank = fill_shape->shape[0].as<IntImmNode>();
  ICHECK(rank) << "Parameter shape must have static rank";

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
  ICHECK_EQ(types.size(), 5);
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

  const auto* begin = types[1].as<TensorTypeNode>();
  if (begin == nullptr) {
    return false;
  }
  ICHECK(begin);

  // calculate output shape
  std::vector<IndexExpr> oshape(num_axis);
  int64_t num_dynamic_axes = begin->shape[0].as<IntImmNode>()->value;
  for (int64_t i = 0; i < num_dynamic_axes; ++i) {
    oshape[i] = Any();
  }

  for (int64_t i = num_dynamic_axes; i < num_axis; ++i) {
    oshape[i] = dshape[i];
  }

  reporter->Assign(types[4], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> StridedSliceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                      const Type& out_type) {
  te::Tensor data = inputs[0];
  te::Tensor begin = inputs[1];
  te::Tensor end = inputs[2];
  te::Tensor strides = inputs[3];
  // Dynamic computation
  int64_t data_rank = data->shape.size();
  int64_t num_dynamic_axes = begin->shape[0].as<IntImmNode>()->value;
  ICHECK(end->shape[0].as<IntImmNode>()->value == num_dynamic_axes &&
         strides->shape[0].as<IntImmNode>()->value == num_dynamic_axes)
      << "begin, end, strides should have the same length if they are dynamic variables";
  ICHECK(num_dynamic_axes <= data_rank)
      << "the number of dynamic axes to slice should be less than or equal to the data rank";
  return Array<te::Tensor>{topi::dynamic_strided_slice(data, begin, end, strides)};
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

bool SparseToDenseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(num_inputs, 4);
  auto sparse_indices = types[0].as<TensorTypeNode>();
  auto sparse_values = types[1].as<TensorTypeNode>();
  auto default_value = types[2].as<TensorTypeNode>();
  auto output_shape = types[3].as<TensorTypeNode>();

  if (sparse_indices == nullptr || sparse_values == nullptr || default_value == nullptr ||
      output_shape == nullptr) {
    return false;
  }

  CHECK(sparse_indices->dtype.is_int()) << "sparse_indices must be tensor of integers";

  CHECK_LE(sparse_indices->shape.size(), 3)
      << "sparse_indices must be a tensor of either 0D, 1D or 2D";

  CHECK_LE(sparse_values->shape.size(), 2) << "sparse_values must be a tensor of either 0D, 1D";

  CHECK_EQ(default_value->shape.size(), 0) << "default_value should be a scalar";

  Array<IndexExpr> oshape;
  for (int i = 0; i < output_shape->shape[0].as<IntImmNode>()->value; i++) {
    oshape.push_back(Any());
  }
  reporter->Assign(types[4], TensorType(oshape, sparse_values->dtype));
  return true;
}

Array<te::Tensor> SparseToDenseCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                       const Type& out_type) {
  ICHECK_EQ(inputs.size(), 4);
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  ICHECK(out_ttype);
  return {topi::sparse_to_dense(inputs[0], out_ttype->shape, inputs[1], inputs[2]())};
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.sparse_to_dense")
    .set_body_typed([](Expr indices, Expr output_shape, Expr values, Expr default_value) {
      static const Op& op = Op::Get("dyn.sparse_to_dense");
      return Call(op, {indices, values, default_value, output_shape});
    });

RELAY_REGISTER_OP("dyn.sparse_to_dense")
    .describe(R"code(A dense tensor from a sparse representation.

    - **sparse_indices**: A 0-D, 1-D, or 2-D tensor of integers containing location of sparse values

    - **output_shape**: A list of integers. Shape of the dense output tensor.

    - **sparse_values**: A 0-D or 1-D tensor containing the sparse values for the sparse indices.

    - **default_value**: A 0-D tensor containing the default value for the remaining locations. Defaults to 0.

    Example::
      -  sparse_to_dense([0, 0], [1, 2]], [3, 4], [1, 2], 0) = [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(4)
    .set_support_level(3)
    .add_argument("sparse_indices", "Tensor", "Contains sparse indices.")
    .add_argument("sparse_values", "Tensor", "Contains values for sparse indices.")
    .add_argument("default_value", "Tensor", "Value to set for non-sparse indices. Defaults to 0.")
    .add_argument("output_shape", "Tensor", "Shape of the dense output tensor")
    .add_type_rel("DynSparseToDense", SparseToDenseRel)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute", SparseToDenseCompute);

/* relay.dyn.unsqueeze */
TVM_REGISTER_NODE_TYPE(DynExpandDimsAttrs);

bool ExpandDimsRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(num_inputs, 2);
  const auto* data_type = types[0].as<TensorTypeNode>();
  if (data_type == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "expand_dims: expect input type to be TensorType but get " << types[0];
    return false;
  }

  const auto* param = attrs.as<DynExpandDimsAttrs>();

  // We don't know the output shape until we see the value of the axis input
  int ndim = data_type->shape.size();
  Array<IndexExpr> oshape(ndim + param->num_newaxis, Any());

  const auto* axis_type = types[1].as<TensorTypeNode>();
  ICHECK(axis_type->shape.size() == 0) << "Axis should be a scalar got shape " << axis_type->shape;

  // Set output shape
  reporter->Assign(types[2], TensorType(oshape, data_type->dtype));
  return true;
}

Array<te::Tensor> ExpandDimsCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                    const Type& out_type) {
  // inputs = [Input tensor, axis to expand]
  ICHECK_EQ(inputs.size(), 2);

  const auto* param = attrs.as<DynExpandDimsAttrs>();

  Array<IndexExpr> ishape = inputs[0]->shape;
  const TensorTypeNode* out_ttype = out_type.as<TensorTypeNode>();
  int ndim_out = out_ttype->shape.size();
  int ndim_in = ishape.size();
  ICHECK_EQ(ndim_in + param->num_newaxis, ndim_out);

  Array<IndexExpr> newshape;
  for (auto val : out_ttype->shape) {
    // These vars will be populated by the VM executor with the results
    // of the shape_func for the op.
    newshape.push_back(val.as<tir::AnyNode>()->ToVar());
  }

  return {topi::reshape(inputs[0], newshape)};
}

Expr MakeExpandDims(Expr data, Expr axis_tensor, int num_newaxis) {
  auto attrs = make_object<DynExpandDimsAttrs>();
  attrs->num_newaxis = num_newaxis;
  static const Op& op = Op::Get("dyn.expand_dims");
  return Call(op, {data, axis_tensor}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.expand_dims").set_body_typed(MakeExpandDims);

RELAY_REGISTER_OP("dyn.expand_dims")
    .describe(R"code(Insert one new axis at the position given by `axis`

- **data**: The input data to the operator.
- **axis**: The axis to insert a new dimension

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("axis", "Tensor", "The axis to insert at a dimension.")
    .set_support_level(3)
    .add_type_rel("DynamicExpandDims", ExpandDimsRel)
    .set_attr<FTVMCompute>("FTVMCompute", ExpandDimsCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

bool DynSqueezeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  // [data, axes, output]
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* axes = types[1].as<TensorTypeNode>();
  if (axes == nullptr) {
    return false;
  }

  ICHECK_EQ(axes->shape.size(), 1) << "Got" << axes->shape.size() << "expected 1";
  ICHECK(axes->shape[0].as<IntImmNode>()) << "axes expected to be static rank";
  size_t output_rank = data->shape.size() - axes->shape[0].as<IntImmNode>()->value;
  std::vector<IndexExpr> result_shape(output_rank, Any());
  reporter->Assign(types[2], TensorType(result_shape, data->dtype));
  return true;
}

Array<te::Tensor> SqueezeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  ICHECK(out_ttype != nullptr);
  Array<IndexExpr> newshape;
  for (auto val : out_ttype->shape) {
    newshape.push_back(val.as<tir::AnyNode>()->ToVar());
  }
  return {topi::reshape(inputs[0], newshape)};
}

Expr MakeDynSqueeze(Expr data, Expr axes) {
  auto attrs = make_object<SqueezeAttrs>();
  static const Op& op = Op::Get("dyn.squeeze");
  return Call(op, {data, axes}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn._make.squeeze").set_body_typed(MakeDynSqueeze);

RELAY_REGISTER_OP("dyn.squeeze")
    .describe(R"code(Remove axes of value 1 in input tensor at the dimensions given by axes

- **data**: The input data to the operator.
- **axes**: The axes to squeeze.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attrs_type<SqueezeAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("axes", "Tensor", "The axes to squeeze.")
    .set_support_level(3)
    .add_type_rel("DynSqueeze", DynSqueezeRel)
    .set_attr<FTVMCompute>("FTVMCompute", SqueezeCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_attr<TReshapeOp>("TReshapeOp", true);

}  // namespace dyn
}  // namespace relay
}  // namespace tvm
