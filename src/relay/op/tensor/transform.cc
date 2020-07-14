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
 * \brief Transform operators.
 */
#include "transform.h"

#include <topi/broadcast.h>
#include <topi/elemwise.h>
#include <topi/nn.h>
#include <topi/reduction.h>
#include <topi/transform.h>
#include <tvm/ir/error.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <vector>

#include "../../transforms/infer_layout_util.h"
#include "../../transforms/pattern_util.h"
#include "../make_op.h"
#include "../op_common.h"

namespace tvm {
namespace relay {
using tir::IntImmNode;

// relay.cast
TVM_REGISTER_NODE_TYPE(CastAttrs);

bool CastRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "cast: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<CastAttrs>();
  reporter->Assign(types[1], TensorType(data->shape, param->dtype));
  return true;
}

Array<te::Tensor> CastCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  const CastAttrs* param = attrs.as<CastAttrs>();
  CHECK(param != nullptr);
  DataType dtype = param->dtype;
  return {topi::cast(inputs[0], dtype)};
}

Expr MakeCast(Expr data, DataType dtype) {
  auto attrs = make_object<CastAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("cast");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.ir.cast").set_body_typed(MakeCast);

RELAY_REGISTER_OP("cast")
    .describe(R"code(Cast the data into a new data type.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<CastAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Cast", CastRel)
    .set_attr<FTVMCompute>("FTVMCompute", CastCompute)
    .set_attr<TOpPattern>("TOpPattern", kElemWise)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

// relay.cast_like
bool CastLikeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "cast: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* dtype_like = types[1].as<TensorTypeNode>();
  if (dtype_like == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
        << "cast: expect input type to be TensorType but get " << types[1];
    return false;
  }
  reporter->Assign(types[2], TensorType(data->shape, dtype_like->dtype));
  return true;
}

Array<te::Tensor> CastLikeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                  const Type& out_type) {
  return {topi::cast(inputs[0], inputs[1]->dtype)};
}

Expr MakeCastLike(Expr data, Expr dtype_like) {
  static const Op& op = Op::Get("cast_like");
  return Call(op, {data, dtype_like}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.ir.cast_like").set_body_typed(MakeCastLike);

RELAY_REGISTER_OP("cast_like")
    .describe(R"code(Cast the data into the type of another tensor.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("dtype_like", "Tensor", "The tensor to cast to.")
    .set_support_level(3)
    .add_type_rel("CastLike", CastLikeRel)
    .set_attr<FTVMCompute>("FTVMCompute", CastLikeCompute)
    .set_attr<TOpPattern>("TOpPattern", kElemWise)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

Array<te::Tensor> ReinterpretCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                     const Type& out_type) {
  const CastAttrs* param = attrs.as<CastAttrs>();
  CHECK(param != nullptr);
  DataType dtype = param->dtype;
  return {topi::reinterpret(inputs[0], dtype)};
}

Expr MakeReinterpret(Expr data, DataType dtype) {
  auto attrs = make_object<CastAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("reinterpret");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay._make.reinterpret").set_body([](const TVMArgs& args, TVMRetValue* rv) {
  runtime::detail::unpack_call<Expr, 2>(MakeReinterpret, args, rv);
});

RELAY_REGISTER_OP("reinterpret")
    .describe(R"code(Reinterpret the data into a new data type.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<CastAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Reinterpret", CastRel)
    .set_attr<FTVMCompute>("FTVMCompute", ReinterpretCompute)
    .set_attr<TOpPattern>("TOpPattern", kElemWise)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

// relay.expand_dims
TVM_REGISTER_NODE_TYPE(ExpandDimsAttrs);

bool ExpandDimsRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "expand_dims: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<ExpandDimsAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  const int axis = param->axis;
  const int num_newaxis = param->num_newaxis;
  CHECK(num_newaxis >= 0) << "expand_dims only accepts `num_newaxis >= 0`"
                          << ", but got num_newaxis = " << num_newaxis;
  CHECK(-ndim - 1 <= axis && axis <= ndim)
      << "expand_dims only accepts `axis` in [-data.ndim - 1, data.ndim]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  const int pivot = axis < 0 ? ndim + axis + 1 : axis;
  std::vector<IndexExpr> oshape;
  oshape.reserve(ndim + num_newaxis);
  for (int i = 0; i < pivot; ++i) {
    oshape.emplace_back(data->shape[i]);
  }
  for (int i = 0; i < num_newaxis; ++i) {
    oshape.emplace_back(1);
  }
  for (int i = pivot; i < ndim; ++i) {
    oshape.emplace_back(data->shape[i]);
  }
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> ExpandDimsCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                    const Type& out_type) {
  const ExpandDimsAttrs* param = attrs.as<ExpandDimsAttrs>();
  CHECK(param != nullptr);
  return {topi::expand_dims(inputs[0], param->axis, param->num_newaxis)};
}

Expr MakeExpandDims(Expr data, int axis, int num_newaxis) {
  auto attrs = make_object<ExpandDimsAttrs>();
  attrs->axis = axis;
  attrs->num_newaxis = num_newaxis;
  static const Op& op = Op::Get("expand_dims");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.expand_dims").set_body_typed(MakeExpandDims);

RELAY_REGISTER_OP("expand_dims")
    .describe(R"code(Insert `num_newaxis` axises at the position given by `axis`

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<ExpandDimsAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(1)
    .add_type_rel("ExpandDims", ExpandDimsRel)
    .set_attr<FTVMCompute>("FTVMCompute", ExpandDimsCompute)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

// relay.concatenate
TVM_REGISTER_NODE_TYPE(ConcatenateAttrs);

Array<te::Tensor> ConcatenateCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                     const Type& out_type) {
  const ConcatenateAttrs* param = attrs.as<ConcatenateAttrs>();
  CHECK(param != nullptr);
  return {topi::concatenate(inputs, param->axis)};
}

Expr MakeConcatenate(Expr data, int axis) {
  auto attrs = make_object<ConcatenateAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("concatenate");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.concatenate").set_body_typed(MakeConcatenate);

RELAY_REGISTER_OP("concatenate")
    .describe(R"code(Concatenate the input tensors along the given axis.

- **data** : A list of tensors.

- **axis** : The axis along which the tensors are concatenated.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ConcatenateAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input list of tensors.")
    .set_support_level(1)
    .add_type_rel("Concatenate", ConcatenateRel<ConcatenateAttrs>)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConcatenateLayout)
    .set_attr<FTVMCompute>("FTVMCompute", ConcatenateCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

TVM_REGISTER_NODE_TYPE(StackAttrs);

bool StackRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
              const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "cast: expect input type to be TupleType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<StackAttrs>();
  const auto& first = Downcast<TensorType>(tensor_tuple->fields[0]);
  const int ndim = static_cast<int>(first->shape.size());

  // Sanity check: axis
  int axis = param->axis;
  CHECK(-ndim <= axis && axis < ndim) << "stack only accepts `axis` in [-ndim, ndim)"
                                      << ", but got axis = " << axis << ", and ndim = " << ndim;
  axis = axis < 0 ? ndim + axis + 1 : axis;

  // Sanity check: ndim and dtype.
  const DataType dtype = first->dtype;
  for (const Type& ele : tensor_tuple->fields) {
    const auto& e = Downcast<TensorType>(ele);
    int e_ndim = static_cast<int>(e->shape.size());
    const DataType& e_dtype = e->dtype;
    CHECK_EQ(e_ndim, ndim) << "relay.stack requires all tensors have the same ndim";
    CHECK_EQ(e_dtype, dtype) << "relay.stack requires all tensors have the same dtype";
    for (size_t j = 0; j < first->shape.size(); ++j) {
      if (j == static_cast<size_t>(axis)) continue;
      if (reporter->AssertEQ(first->shape[j], e->shape[j])) continue;
      throw Error(
          "relay.stack requires all tensors have the same shape "
          "on non-stacking axes");
    }
  }

  // Calculate shape
  std::vector<IndexExpr> oshape;
  oshape.reserve(ndim + 1);
  const int stack_dim = static_cast<int>(tensor_tuple->fields.size());
  for (int i = 0; i < axis; ++i) {
    oshape.emplace_back(first->shape[i]);
  }
  oshape.emplace_back(stack_dim);
  for (int i = axis; i < ndim; ++i) {
    oshape.emplace_back(first->shape[i]);
  }
  reporter->Assign(types[1], TensorType(oshape, dtype));
  return true;
}

Array<te::Tensor> StackCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                               const Type& out_type) {
  const StackAttrs* param = attrs.as<StackAttrs>();
  CHECK(param != nullptr);
  return {topi::stack(inputs, param->axis)};
}

Expr MakeStack(Expr data, int axis) {
  auto attrs = make_object<StackAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("stack");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.stack").set_body_typed(MakeStack);

RELAY_REGISTER_OP("stack")
    .describe(R"code(Stack the input tensors along the given axis.

- **data** : A list of tensors.

- **axis** : The axis along which the tensors are stacked.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<StackAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input list of tensors.")
    .set_support_level(3)
    .add_type_rel("Stack", StackRel)
    .set_attr<FTVMCompute>("FTVMCompute", StackCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

/* relay.transpose */
TVM_REGISTER_NODE_TYPE(TransposeAttrs);

bool TransposeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "transpose: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<TransposeAttrs>();
  const int ndim = data->shape.size();
  const Array<Integer>& axes = param->axes;
  // check dimension match
  CHECK(!axes.defined() || static_cast<int>(axes.size()) == ndim)
      << "Dimension mismatch: axes has " << axes.size() << " elements"
      << ", but data.ndim = " << ndim;
  // construct int_axes
  std::vector<int> int_axes;
  int_axes.reserve(ndim);
  // used not defined to check if it is None.
  if (!axes.defined()) {
    for (int i = ndim - 1; i >= 0; --i) {
      int_axes.push_back(i);
    }
  } else {
    std::vector<int> axis_used(ndim, 0);
    for (const Integer& e : axes) {
      int64_t axis = e;
      // sanity check for axis and ndim
      CHECK(-ndim <= axis && axis < ndim)
          << "transpose only allows each `axis` in `axes` in range [-data.ndim, data.ndim)"
          << ", but got axis = " << axis << ", and data.ndim = " << ndim;
      axis = axis < 0 ? axis + ndim : axis;
      // sanity check for duplication
      CHECK(!axis_used[axis]) << "Duplicate axes in transpose: " << axis;
      axis_used[axis] = 1;
      int_axes.push_back(static_cast<int>(axis));
    }
  }
  std::vector<IndexExpr> oshape;
  oshape.reserve(ndim);
  for (int axis : int_axes) {
    oshape.push_back(data->shape[axis]);
  }
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> TransposeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                   const Type& out_type) {
  const auto* param = attrs.as<TransposeAttrs>();
  CHECK(param != nullptr);
  return Array<te::Tensor>{topi::transpose(inputs[0], param->axes)};
}

Expr MakeTranspose(Expr data, Array<Integer> axes) {
  auto attrs = make_object<TransposeAttrs>();
  attrs->axes = std::move(axes);
  static const Op& op = Op::Get("transpose");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.transpose").set_body_typed(MakeTranspose);

RELAY_REGISTER_OP("transpose")
    .describe(R"code(Permutes the dimensions of an array.

- **data**: The input data to the operator.

- **axes**: The target axes order, reverse order if not specified.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<TransposeAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Transpose", TransposeRel)
    .set_attr<FTVMCompute>("FTVMCompute", TransposeCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

/* relay.reshape */
TVM_REGISTER_NODE_TYPE(ReshapeAttrs);

bool ReshapeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  const auto* param = attrs.as<ReshapeAttrs>();
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "reshape: expect input type to be TensorType but get " << types[0];
    return false;
  }

  Array<IndexExpr> oshape;
  Array<IndexExpr> data_shape;
  Array<Integer> newshape;

  if (param->reverse) {
    data_shape.Assign(data->shape.rbegin(), data->shape.rend());
    newshape.Assign(param->newshape.rbegin(), param->newshape.rend());
  } else {
    data_shape = data->shape;
    newshape = param->newshape;
  }

  std::unordered_set<size_t> used_input_dims;
  std::unordered_set<size_t> used_output_dims;
  size_t src_idx = 0;
  int infer_idx = -1;

  for (size_t i = 0; i < newshape.size(); ++i) {
    int svalue = newshape[i]->value;
    // special flag handling for shape inference.
    if (svalue > 0) {
      oshape.push_back(newshape[i]);
      ++src_idx;
    } else if (svalue == 0) {
      // keep same
      CHECK_LT(src_idx, data_shape.size());
      used_input_dims.insert(src_idx);
      used_output_dims.insert(oshape.size());
      oshape.push_back(data_shape[src_idx++]);
    } else if (svalue == -1) {
      // inference based on rest
      CHECK_LT(infer_idx, 0) << "One and only one dim can be inferred";
      infer_idx = i;
      oshape.push_back(1);
      ++src_idx;
    } else if (svalue == -2) {
      // copy all remaining dims from source
      while (src_idx < data_shape.size()) {
        used_input_dims.insert(src_idx);
        used_output_dims.insert(oshape.size());
        oshape.push_back(data_shape[src_idx++]);
      }
    } else if (svalue == -3) {
      // merge two dims from source
      CHECK_LT(src_idx + 1, data_shape.size());
      used_input_dims.insert(src_idx);
      IndexExpr d1 = data_shape[src_idx++];
      used_input_dims.insert(src_idx);
      IndexExpr d2 = data_shape[src_idx++];
      used_output_dims.insert(oshape.size());
      if (d1.as<AnyNode>() || d2.as<AnyNode>()) {
        oshape.push_back(Any());
      } else {
        oshape.push_back(d1 * d2);
      }
    } else if (svalue == -4) {
      // split the source dim s into two dims
      // read the left dim and then the right dim (either can be -1)
      CHECK_LT(i + 2, newshape.size());
      CHECK_LT(src_idx, data_shape.size());
      used_input_dims.insert(src_idx);
      IndexExpr d0 = data_shape[src_idx++];
      Integer d1 = newshape[++i];
      Integer d2 = newshape[++i];
      if (d1->value == -1) {
        CHECK(d2->value != -1) << "Split dims cannot both be -1.";
        used_output_dims.insert(oshape.size());
        if (d0.as<AnyNode>()) {
          oshape.push_back(Any());
        } else {
          oshape.push_back(indexdiv(d0, d2));
        }
        used_output_dims.insert(oshape.size());
        oshape.push_back(d2);
      } else {
        used_output_dims.insert(oshape.size());
        oshape.push_back(d1);
        used_output_dims.insert(oshape.size());
        if (d2->value == -1) {
          if (d0.as<AnyNode>()) {
            oshape.push_back(Any());
          } else {
            oshape.push_back(indexdiv(d0, d1));
          }
        } else {
          oshape.push_back(d2);
        }
      }
    } else {
      CHECK(false) << "Unsupported special value: " << svalue;
    }
  }

  if (infer_idx >= 0) {
    IndexExpr infer_dim = 1;
    for (size_t i = 0; i < data_shape.size(); ++i) {
      if (used_input_dims.count(i) != 0) {
        continue;
      }
      if (data_shape[i].as<AnyNode>()) {
        infer_dim = Any();
        break;
      }
      infer_dim *= data_shape[i];
    }
    if (!infer_dim.as<AnyNode>()) {
      for (size_t i = 0; i < oshape.size(); ++i) {
        if (used_output_dims.count(i) != 0) {
          continue;
        }
        if (oshape[i].as<AnyNode>()) {
          infer_dim = Any();
          break;
        }
        infer_dim = indexdiv(infer_dim, oshape[i]);
      }
    }
    oshape.Set(infer_idx, infer_dim);
  }

  if (param->reverse) {
    reporter->Assign(types[1],
                     TensorType(Array<IndexExpr>(oshape.rbegin(), oshape.rend()), data->dtype));
  } else {
    reporter->Assign(types[1], TensorType(oshape, data->dtype));
  }
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

Expr MakeReshape(Expr data, Array<Integer> newshape) {
  auto attrs = make_object<ReshapeAttrs>();
  attrs->newshape = std::move(newshape);
  attrs->reverse = false;
  static const Op& op = Op::Get("reshape");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.reshape").set_body_typed(MakeReshape);

RELAY_REGISTER_OP("reshape")
    .describe(R"code(Reshapes the input array.

Example::

To give user more convenience in without doing manual shape inference,
some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}.
The significance of each is explained below:

- ``0``  copy this dimension from the input to the output shape.

Example::

- data.shape = (2,3,4), newshape = (4,0,2), result.shape = (4,3,2)
- data.shape = (2,3,4), newshape = (2,0,0), result.shape = (2,3,4)

- ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
keeping the size of the new array same as that of the input array.
At most one dimension of shape can be -1.

Example::

- data.shape = (2,3,4), newshape = (6,1,-1), result.shape = (6,1,4)
- data.shape = (2,3,4), newshape = (3,-1,8), result.shape = (3,1,8)
- data.shape = (2,3,4), newshape = (-1,), result.shape = (24,)

- ``-2`` copy all/remainder of the input dimensions to the output shape.

Example::

- data.shape = (2,3,4), newshape = (-2,), result.shape = (2,3,4)
- data.shape = (2,3,4), newshape = (2,-2), result.shape = (2,3,4)
- data.shape = (2,3,4), newshape = (-2,1,1), result.shape = (2,3,4,1,1)

- ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.

Example::

- data.shape = (2,3,4), newshape = (-3,4), result.shape = (6,4)
- data.shape = (2,3,4,5), newshape = (-3,-3), result.shape = (6,20)
- data.shape = (2,3,4), newshape = (0,-3), result.shape = (2,12)
- data.shape = (2,3,4), newshape = (-3,-2), result.shape = (6,4)

- ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).

Example::

- data.shape = (2,3,4), newshape = (-4,1,2,-2), result.shape =(1,2,3,4)
- data.shape = (2,3,4), newshape = (2,-4,-1,3,-2), result.shape = (2,1,3,4)

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<ReshapeAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Reshape", ReshapeRel)
    .set_attr<FTVMCompute>("FTVMCompute", ReshapeCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

/*!
 * \brief ReshapeLikeRel User defined type constraint function.
 * \param num_inputs Number of input types in the args.
 * \param attrs The additional attributes of the operator.
 * \param reporter The reporter to report solution to.
 * \return False if the relation has not been resolved, it might be resolved later.
 *  True if this relation has been resolved.
 */
bool ReshapeLikeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* reshape_like = types[1].as<TensorTypeNode>();
  if (reshape_like == nullptr) {
    return false;
  }
  // Only check When input data has static shape.
  bool is_static_shape = true;
  for (size_t i = 0; i < data->shape.size(); ++i) {
    if (!data->shape[i].as<IntImmNode>()) {
      is_static_shape = false;
      break;
    }
  }
  if (is_static_shape) {
    CHECK(reporter->AssertEQ(data->Size(), reshape_like->Size()))
        << "Reshape inputs size should be compatible.";
  }
  reporter->Assign(types[2], TensorType(reshape_like->shape, data->dtype));
  return true;
}

Expr MakeReshapeLike(Expr data, Expr shape_like) {
  static const Op& op = Op::Get("reshape_like");
  return Call(op, {data, shape_like}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.reshape_like").set_body_typed(MakeReshapeLike);

RELAY_REGISTER_OP("reshape_like")
    .describe(R"code(Reshapes the input array by the size of another array.
For an input array with shape ``(d1, d2, ..., dk)``, `reshape_like` operation reshapes
the input array into an output array with the same shape as the second input array.
.. note::
    Sizes for both array should be compatible.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape_like", "Tensor", "Shape tensor.")
    .set_support_level(3)
    .add_type_rel("ReshapeLike", ReshapeLikeRel)
    .set_attr<FTVMCompute>("FTVMCompute", ReshapeCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// ArgWhere
bool ArgWhereRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  CHECK_EQ(num_inputs, 1);
  auto tt = types[0].as<TensorTypeNode>();
  CHECK(tt != nullptr);
  const auto& input_shape = tt->shape;
  const auto& input_rank = input_shape.size();
  std::vector<IndexExpr> result_shape;
  result_shape.push_back(Any());
  result_shape.push_back(IntImm(DataType::Int(32), input_rank));
  reporter->Assign(types[1], TensorType(result_shape, DataType::Int(32)));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op._make.argwhere").set_body_typed([](Expr data) {
  static const Op& op = Op::Get("argwhere");
  return Call(op, {data}, Attrs(), {});
});

RELAY_REGISTER_OP("argwhere")
    .describe(R"doc(Find the indices of elements of a tensor that are
non-zero)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("condition", "Tensor", "The input condition tensor.")
    .add_type_rel("ArgWhere", ArgWhereRel)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_support_level(10);

// Scatter
TVM_REGISTER_NODE_TYPE(ScatterAttrs);

// Scatter
bool ScatterRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  CHECK_EQ(num_inputs, 3);
  CHECK_EQ(types.size(), 4);
  auto data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  auto indices = types[1].as<TensorTypeNode>();
  if (indices == nullptr) {
    return false;
  }
  auto updates = types[2].as<TensorTypeNode>();
  if (updates == nullptr) {
    return false;
  }
  CHECK(indices->dtype.is_int()) << "indices of take must be tensor of integer";
  const auto param = attrs.as<ScatterAttrs>();
  CHECK(param != nullptr);
  reporter->Assign(types[3], TensorType(data->shape, data->dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op._make.scatter")
    .set_body_typed([](Expr data, Expr indices, Expr updates, int axis) {
      auto attrs = make_object<ScatterAttrs>();
      attrs->axis = std::move(axis);
      static const Op& op = Op::Get("scatter");
      return Call(op, {data, indices, updates}, Attrs(attrs), {});
    });

RELAY_REGISTER_OP("scatter")
    .describe(
        R"doc(Update data at positions defined by indices with values in updates)doc" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input data tensor.")
    .add_argument("indicies", "Tensor", "The indicies location tensor.")
    .add_argument("updates", "Tensor", "The values to update the input with.")
    .add_type_rel("Scatter", ScatterRel)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_support_level(10);

// Take
TVM_REGISTER_NODE_TYPE(TakeAttrs);

bool TakeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  // `types` contains: [data, indices, result]
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* indices = types[1].as<TensorTypeNode>();
  if (indices == nullptr) {
    return false;
  }
  CHECK(indices->dtype.is_int()) << "indices of take must be tensor of integer";
  const auto param = attrs.as<TakeAttrs>();
  CHECK(param != nullptr);

  if (!param->axis.defined()) {
    std::vector<IndexExpr> oshape(indices->shape.begin(), indices->shape.end());
    reporter->Assign(types[2], TensorType(oshape, data->dtype));
    return true;
  }

  std::vector<IndexExpr> oshape;
  const auto ndim_data = static_cast<int>(data->shape.size());
  const auto ndim_indices = static_cast<int>(indices->shape.size());
  int axis = static_cast<int>(param->axis->value);
  if (axis < 0) axis += ndim_data;
  CHECK_LE(axis, ndim_data) << "axis should be with in data shape"
                            << ", but got = " << axis;

  oshape.reserve(ndim_data - 1 + ndim_indices);
  for (int i = 0; i < axis; ++i) {
    oshape.emplace_back(data->shape[i]);
  }
  for (int i = 0; i < ndim_indices; ++i) {
    oshape.emplace_back(indices->shape[i]);
  }
  for (int i = axis + 1; i < ndim_data; ++i) {
    oshape.emplace_back(data->shape[i]);
  }

  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> TakeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  const auto* param = attrs.as<TakeAttrs>();
  CHECK(param != nullptr);
  if (!param->axis.defined()) {
    return Array<te::Tensor>{topi::take(inputs[0], inputs[1], param->mode)};
  } else {
    return Array<te::Tensor>{topi::take(inputs[0], inputs[1], param->axis, param->mode)};
  }
}

Expr MakeTake(Expr data, Expr indices, Integer axis, String mode) {
  auto attrs = make_object<TakeAttrs>();
  attrs->axis = std::move(axis);
  attrs->mode = std::move(mode);
  static const Op& op = Op::Get("take");
  return Call(op, {data, indices}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.take").set_body_typed(MakeTake);

RELAY_REGISTER_OP("take")
    .describe(R"code(Take elements from an array along an axis.

When axis is not None, this function does the same thing as 'fancy' indexing
(indexing arrays using arrays); however, it can be easier to use if you need
elements along a given axis.

**Note** that when axis is none the flattened input array is used.

Examples::

  a = [[ 1, 2],
       [ 3, 4]]
  indices = [3, 0, 2]
  take(a, indices) = [ 4, 1, 3]

  a = [[ 1., 2.],
       [ 3., 4.]]
  indices = [1, 0]
  take(a, indices, axis=1) = [[ 2., 1.],
                              [ 4., 3.]]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<TakeAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices tensor.")
    .set_support_level(3)
    .add_type_rel("Take", TakeRel)
    .set_attr<FTVMCompute>("FTVMCompute", TakeCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// Init ops
TVM_REGISTER_NODE_TYPE(InitOpAttrs);

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

  const IntImmNode* shape_shape = fill_shape->shape[0].as<IntImmNode>();
  CHECK(shape_shape) << "Parameter shape must have static shape";

  std::vector<IndexExpr> oshape;
  if (param->shape) {
    const Array<Integer>& cshape_array = param->shape.value();
    for (size_t i = 0; i < cshape_array.size(); ++i) {
      oshape.push_back(cshape_array[i]);
    }
  } else {
    for (int i = 0; i < shape_shape->value; ++i) {
      oshape.push_back(Any());
    }
  }
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

Array<te::Tensor> FullCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  return {topi::full(out_ttype->shape, out_ttype->dtype, inputs[0]())};
}

Expr MakeFull(Expr fill_value, Expr shape, DataType dtype) {
  auto attrs = make_object<InitOpAttrs>();
  if (const auto* cshape = shape.as<ConstantNode>()) {
    attrs->shape = ToVector(cshape->data);
  }
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("full");
  return Call(op, {fill_value, shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.full").set_body_typed(MakeFull);

RELAY_REGISTER_OP("full")
    .describe(R"code(Fill array with scalar value.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<InitOpAttrs>()
    .set_num_inputs(2)
    .add_argument("fill_value", "double", "The value to fill.")
    .add_argument("shape", "Tensor", "Target shape.")
    .set_support_level(3)
    .add_type_rel("Full", FullRel)
    .set_attr<FTVMCompute>("FTVMCompute", FullCompute)
    .set_attr<TOpPattern>("TOpPattern", kElemWise);

bool InitOpRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  const auto* fill_shape = types[0].as<TensorTypeNode>();
  DataType out_dtype = param->dtype;

  const IntImmNode* shape_shape = fill_shape->shape[0].as<IntImmNode>();
  CHECK(shape_shape) << "Parameter shape must have static shape";

  std::vector<IndexExpr> oshape;
  if (param->shape) {
    const Array<Integer>& cshape_array = param->shape.value();
    for (size_t i = 0; i < cshape_array.size(); ++i) {
      oshape.push_back(cshape_array[i]);
    }
  } else {
    for (int i = 0; i < shape_shape->value; ++i) {
      oshape.push_back(Any());
    }
  }
  reporter->Assign(types[1], TensorType(oshape, out_dtype));
  return true;
}

Expr MakeZeros(Expr shape, DataType dtype) {
  auto attrs = make_object<InitOpAttrs>();
  if (const auto* cshape = shape.as<ConstantNode>()) {
    attrs->shape = ToVector(cshape->data);
  }
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("zeros");
  return Call(op, {shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.zeros").set_body_typed(MakeZeros);

RELAY_REGISTER_OP("zeros")
    .describe(R"code(Fill array with zeros.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<InitOpAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Tensor", "Target shape.")
    .set_support_level(3)
    .add_type_rel("InitOp", InitOpRel);

Expr MakeOnes(Expr shape, DataType dtype) {
  auto attrs = make_object<InitOpAttrs>();
  if (const auto* cshape = shape.as<ConstantNode>()) {
    attrs->shape = ToVector(cshape->data);
  }
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("ones");
  return Call(op, {shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.ones").set_body_typed(MakeOnes);

RELAY_REGISTER_OP("ones")
    .describe(R"code(Fill array with ones.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<InitOpAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Tensor", "Target shape.")
    .set_support_level(3)
    .add_type_rel("InitOp", InitOpRel);

bool FullLikeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* fill_value = types[1].as<TensorTypeNode>();
  if (fill_value == nullptr) {
    return false;
  }

  CHECK_EQ(fill_value->shape.size(), 0)
      << "The fill value should be a scalar but here it has dimension " << fill_value->shape.size()
      << ".";

  reporter->Assign(types[2], TensorType(data->shape, data->dtype));
  return true;
}

Array<te::Tensor> FullLikeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                  const Type& out_type) {
  return {topi::full_like(inputs[0], inputs[1]())};
}

Expr MakeFullLike(Expr data, Expr fill_value) {
  static const Op& op = Op::Get("full_like");
  return Call(op, {data, fill_value}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.full_like").set_body_typed(MakeFullLike);

RELAY_REGISTER_OP("full_like")
    .describe(R"code(Return an scalar value array with the same shape
and type as the input array.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("fill_value", "double", "Scalar value to fill.")
    .set_support_level(3)
    .add_type_rel("FullLike", FullLikeRel)
    .set_attr<FTVMCompute>("FTVMCompute", FullLikeCompute)
    .set_attr<TOpPattern>("TOpPattern", kElemWise);

// arange operator
TVM_REGISTER_NODE_TYPE(ArangeAttrs);

bool ArangeRel(const Array<Type>& types, int num_inputs, const Attrs& raw_attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);
  const ArangeAttrs* attrs = raw_attrs.as<ArangeAttrs>();
  const ConstantNode *cstart, *cstop, *cstep;

  reporter->Assign(types[0], types[1]);
  reporter->Assign(types[1], types[2]);
  reporter->Assign(types[2], TensorType({}, attrs->dtype));

  if ((cstart = attrs->start.as<ConstantNode>()) && (cstop = attrs->stop.as<ConstantNode>()) &&
      (cstep = attrs->step.as<ConstantNode>())) {
    double start = ToScalar(cstart->data);
    double stop = ToScalar(cstop->data);
    double step = ToScalar(cstep->data);
    int32_t num_elem = static_cast<int32_t>(std::ceil((stop - start) / step));
    CHECK_GT(num_elem, 0) << "Invalid arange attributes (start, stop, step): " << attrs->start
                          << ", " << attrs->stop << ", " << attrs->step;
    reporter->Assign(types[3], TensorType({num_elem}, attrs->dtype));
    return true;
  } else {
    reporter->Assign(types[3], TensorType({Any()}, attrs->dtype));
    return true;
  }
}

inline te::Tensor DynamicArange(const te::Tensor& start, const te::Tensor& stop,
                                const te::Tensor& step, tvm::DataType dtype,
                                std::string name = "T_arange_dynamic",
                                std::string tag = topi::kInjective) {
  tvm::PrimExpr num_elem = tvm::tir::Var("num_elem");
  return te::compute(
      {num_elem},
      [&](const Array<tvm::tir::Var>& indices) {
        return tvm::cast(dtype, start[0] + step[0] * indices[0]);
      },
      name, tag);
}

Array<te::Tensor> ArangeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  const ArangeAttrs* param = attrs.as<ArangeAttrs>();
  CHECK(param != nullptr);
  te::Tensor start = inputs[0];
  te::Tensor stop = inputs[1];
  te::Tensor step = inputs[2];
  return {DynamicArange(start, stop, step, param->dtype)};
}

Expr MakeArange(Expr start, Expr stop, Expr step, DataType dtype) {
  auto attrs = make_object<ArangeAttrs>();
  attrs->start = start;
  attrs->stop = stop;
  attrs->step = step;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("arange");
  return Call(op, {start, stop, step}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.arange").set_body_typed(MakeArange);

// An issue with the existing design is that we require dependency
// to type the operator precisely.
//
// Supporting this in general is challenging so we duplicate the
// secondary arguments as args and attributes.
//
// In this way reify the arguments at both the value and type level.
//
// In the case our arguments are constant we can immediately recover
// the type of arange.
//
// In general I think we should avoid this pattern, and introduce
// a secondary shape analysis to recover more precise information.
RELAY_REGISTER_OP("arange")
    .describe(R"code(Returns evenly spaced values within a given interval.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ArangeAttrs>()
    .set_num_inputs(3)
    .set_support_level(3)
    .add_type_rel("Arange", ArangeRel)
    .set_attr<FTVMCompute>("FTVMCompute", ArangeCompute)
    // TODO(@icemelon): Change arange to kOpaque because FuseOps doesn't consider dynamic shape
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<AnyCodegenStrategy>("AnyCodegenStrategy", kVariableDimensions);

// repeat operator
TVM_REGISTER_NODE_TYPE(RepeatAttrs);

bool RepeatRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "repeat: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<RepeatAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  const int repeats = param->repeats;
  const int axis = param->axis;
  CHECK(repeats >= 1) << "repeat only accepts `repeats >= 1`"
                      << ", but got repeats = " << repeats;
  CHECK(-ndim - 1 <= axis && axis <= ndim)
      << "repeat only accepts `axis` in [-data.ndim - 1, data.ndim]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  const int pivot = axis < 0 ? ndim + axis : axis;
  std::vector<IndexExpr> oshape;
  oshape.reserve(ndim + repeats);
  for (int i = 0; i < pivot; ++i) {
    oshape.emplace_back(data->shape[i]);
  }
  oshape.emplace_back(data->shape[pivot] * repeats);
  for (int i = pivot + 1; i < ndim; ++i) {
    oshape.emplace_back(data->shape[i]);
  }
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> RepeatCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  const RepeatAttrs* param = attrs.as<RepeatAttrs>();
  CHECK(param != nullptr);
  return {topi::repeat(inputs[0], param->repeats, param->axis)};
}

Expr MakeRepeat(Expr data, int repeats, int axis) {
  auto attrs = make_object<RepeatAttrs>();
  attrs->repeats = repeats;
  attrs->axis = axis;
  static const Op& op = Op::Get("repeat");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.repeat").set_body_typed(MakeRepeat);

RELAY_REGISTER_OP("repeat")
    .describe(R"code(Repeat elements of an array `repeats` times along axis `axis`

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<RepeatAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Repeat", RepeatRel)
    .set_attr<FTVMCompute>("FTVMCompute", RepeatCompute)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

// meshgrid operator
TVM_REGISTER_NODE_TYPE(MeshgridAttrs);

bool MeshgridRel(const Array<Type>& types, int num_inputs, const Attrs& raw_attrs,
                 const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const MeshgridAttrs* attrs = raw_attrs.as<MeshgridAttrs>();
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    throw Error(
        ErrorBuilder() << "meshgrid requires a tuple of tensors as the first argument, found "
                       << PrettyPrint(types[0]));
  } else if (types[0].as<IncompleteTypeNode>() != nullptr) {
    return false;
  }
  const int data_length = static_cast<int>(tensor_tuple->fields.size());

  // Get first dtype.
  const auto& first = Downcast<TensorType>(tensor_tuple->fields[0]);
  const DataType dtype = first->dtype;

  // Get size of output grid.
  std::vector<IndexExpr> grid_shape;
  grid_shape.reserve(data_length);
  for (const Type& ele : tensor_tuple->fields) {
    if (ele.as<IncompleteTypeNode>()) {
      return false;
    }
    const auto& e = Downcast<TensorType>(ele);
    int e_ndim = static_cast<int>(e->shape.size());
    const DataType& e_dtype = e->dtype;
    if (e_dtype != dtype) {
      throw Error("relay.meshgrid requires all tensors have the same dtype");
    }
    if (e_ndim == 0) {
      grid_shape.emplace_back(1);
    } else if (e_ndim == 1) {
      grid_shape.emplace_back(e->shape[0]);
    } else {
      throw Error("relay.meshgrid requires all tensors be either scalars or 1-D vectors.");
    }
  }

  // "xy" mode swaps first two dimensions
  if (attrs->indexing == "xy" && grid_shape.size() >= 2) {
    std::swap(grid_shape[0], grid_shape[1]);
  }

  // There is one output grid for each input, all with same shape.
  std::vector<Type> grids;
  grids.reserve(data_length);
  for (int i = 0; i < data_length; i++) {
    grids.emplace_back(TensorType(grid_shape, dtype));
  }
  reporter->Assign(types[1], TupleType(Array<Type>(grids)));
  return true;
}

Array<te::Tensor> MeshgridCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                  const Type& out_type) {
  const MeshgridAttrs* param = attrs.as<MeshgridAttrs>();
  CHECK(param != nullptr);
  return {topi::meshgrid(inputs, param->indexing)};
}

Expr MakeMeshgrid(Expr data, String indexing) {
  auto attrs = make_object<MeshgridAttrs>();
  attrs->indexing = std::move(indexing);
  static const Op& op = Op::Get("meshgrid");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.meshgrid").set_body_typed(MakeMeshgrid);

RELAY_REGISTER_OP("meshgrid")
    .describe(R"code(Create coordinate matrices from coordinate vectors.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<MeshgridAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input list of tensors.")
    .set_support_level(3)
    .add_type_rel("Meshgrid", MeshgridRel)
    .set_attr<FTVMCompute>("FTVMCompute", MeshgridCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// tile operator
TVM_REGISTER_NODE_TYPE(TileAttrs);

bool TileRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "tile: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<TileAttrs>();
  const size_t ndim = data->shape.size();
  const Array<Integer>& reps = param->reps;
  // check dimension match
  CHECK(reps.defined()) << "repetition array is not defined. data.ndim = " << ndim;
  const size_t rndim = reps.size();
  for (size_t i = 0; i < rndim; ++i) {
    if (const tvm::tir::IntImmNode* val = reps[i].as<tvm::tir::IntImmNode>()) {
      CHECK_GT(val->value, 0) << "Tile reps value should always be larger than 0, but get: "
                              << val->value;
    }
  }
  size_t tndim = (ndim > rndim) ? ndim : rndim;
  // re-construct data shape or reps shape
  std::vector<IndexExpr> data_shape;
  std::vector<IndexExpr> reps_shape;
  data_shape.reserve(tndim);
  reps_shape.reserve(tndim);
  if (ndim == rndim) {
    for (size_t i = 0; i < tndim; ++i) {
      data_shape.emplace_back(data->shape[i]);
      reps_shape.emplace_back(reps[i]);
    }
  } else if (ndim > rndim) {
    for (size_t i = 0; i < ndim; ++i) {
      data_shape.emplace_back(data->shape[i]);
    }
    for (size_t i = 0; i < (ndim - rndim); ++i) {
      reps_shape.emplace_back(1);
    }
    for (size_t i = 0; i < rndim; ++i) {
      reps_shape.emplace_back(reps[i]);
    }
  } else {
    for (size_t i = 0; i < rndim; ++i) {
      reps_shape.emplace_back(reps[i]);
    }
    for (size_t i = 0; i < (rndim - ndim); ++i) {
      data_shape.emplace_back(1);
    }
    for (size_t i = 0; i < ndim; ++i) {
      data_shape.emplace_back(data->shape[i]);
    }
  }
  std::vector<IndexExpr> oshape;
  oshape.reserve(tndim);
  for (size_t i = 0; i < tndim; ++i) {
    // Save Any if it is dynamic shape
    if (!data_shape[i].as<IntImmNode>()) {
      oshape.emplace_back(Any());
    } else {
      oshape.emplace_back(data_shape[i] * reps_shape[i]);
    }
  }
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> TileCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  const TileAttrs* param = attrs.as<TileAttrs>();
  CHECK(param != nullptr);
  return {topi::tile(inputs[0], param->reps)};
}

Expr MakeTile(Expr data, Array<Integer> reps) {
  auto attrs = make_object<TileAttrs>();
  attrs->reps = reps;
  static const Op& op = Op::Get("tile");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.tile").set_body_typed(MakeTile);

RELAY_REGISTER_OP("tile")
    .describe(R"code(Repeat the whole array multiple times.

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<TileAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Tile", TileRel)
    .set_attr<FTVMCompute>("FTVMCompute", TileCompute)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

// reverse operator
TVM_REGISTER_NODE_TYPE(ReverseAttrs);

bool ReverseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "reverse: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<ReverseAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  const int axis = param->axis;
  CHECK(-ndim <= axis && axis < ndim)
      << "reverse only accepts `axis` in [-data.ndim, data.ndim - 1]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  reporter->Assign(types[1], types[0]);
  return true;
}

Array<te::Tensor> ReverseCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {
  const ReverseAttrs* param = attrs.as<ReverseAttrs>();
  CHECK(param != nullptr);
  // pass empty seq_length tensor to reverse_sequence
  return {topi::reverse_sequence(inputs[0], te::Tensor(), param->axis)};
}

Expr MakeReverse(Expr data, int axis) {
  auto attrs = make_object<ReverseAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("reverse");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.reverse").set_body_typed(MakeReverse);

RELAY_REGISTER_OP("reverse")
    .describe(R"code(Reverses the order of elements along given `axis` while preserving array shape.

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<ReverseAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Reverse", ReverseRel)
    .set_attr<FTVMCompute>("FTVMCompute", ReverseCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// reverse sequence operator
TVM_REGISTER_NODE_TYPE(ReverseSequenceAttrs);

bool ReverseSequenceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  // `types` contains: [data, seq_lengths, result]
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "reverse_sequence: expect input type to be TensorType but get " << types[0];
    return false;
  }

  const auto* seq_lengths = types[1].as<TensorTypeNode>();
  if (seq_lengths == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
        << "reverse_sequence: expect input type to be TensorType but get " << types[1];
    return false;
  }

  const int seq_lengths_dim = static_cast<int>(seq_lengths->shape.size());
  CHECK(seq_lengths_dim == 1) << "For reverse_sequnece, seq_lengths must be a 1D vector";
  CHECK(seq_lengths->dtype.is_int())
      << "For reverse_sequnece, seq_lengths must be tensor of integer";

  const auto* param = attrs.as<ReverseSequenceAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  int batch_axis = param->batch_axis;
  CHECK(-ndim <= batch_axis && batch_axis < ndim)
      << "reverse_sequence only accepts `batch_axis` in [-data.ndim, data.ndim - 1]"
      << ", but got batch_axis = " << batch_axis << ", and data.ndim = " << ndim;

  if (batch_axis < 0) {
    batch_axis = static_cast<int>(data->shape.size()) + batch_axis;
  }
  CHECK(reporter->Assert(seq_lengths->shape[0] == data->shape[batch_axis]))
      << "For reverse_sequnece seq_lengths size should match with dimension of batch axis"
      << ", but got dimension of batch_axis = " << data->shape[batch_axis]
      << ", and seq_length size = " << seq_lengths->shape[0];

  const int seq_axis = param->seq_axis;
  CHECK(-ndim <= seq_axis && seq_axis < ndim)
      << "reverse_sequnece only accepts `seq_axis` in [-data.ndim, data.ndim - 1]"
      << ", but got seq_axis = " << seq_axis << ", and data.ndim = " << ndim;

  reporter->Assign(types[2], types[0]);
  return true;
}

Array<te::Tensor> ReverseSequenceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                         const Type& out_type) {
  const ReverseSequenceAttrs* param = attrs.as<ReverseSequenceAttrs>();
  CHECK(param != nullptr);
  return {topi::reverse_sequence(inputs[0], inputs[1], param->seq_axis, param->batch_axis)};
}

Expr MakeReverseSequence(Expr data, Expr seq_lengths, int seq_axis, int batch_axis) {
  auto attrs = make_object<ReverseSequenceAttrs>();
  attrs->seq_axis = seq_axis;
  attrs->batch_axis = batch_axis;
  static const Op& op = Op::Get("reverse_sequence");
  return Call(op, {data, seq_lengths}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.reverse_sequence").set_body_typed(MakeReverseSequence);

RELAY_REGISTER_OP("reverse_sequence")
    .describe(R"code(Reverses the tensor for variable length slices.
Input is first sliced along batch axis and then elements are reversed along seq axis.

- **data**: The input data to the operator.

- **seq_lengths**: A 1D Tensor with length data.dims[batch_axis].

- **seq_axis**: The axis along which the elements will be reversed. Default is 1.

- **batch_axis**: The axis along which the tensor will be sliced. Default is 0.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attrs_type<ReverseSequenceAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("seq_lengths", "Tensor", "A 1D Tensor with length data.dims[batch_axis]")
    .set_support_level(3)
    .add_type_rel("ReverseSequence", ReverseSequenceRel)
    .set_attr<FTVMCompute>("FTVMCompute", ReverseSequenceCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// where operator
bool WhereRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
              const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4U);
  const auto* condition = types[0].as<TensorTypeNode>();
  const auto* x = types[1].as<TensorTypeNode>();
  const auto* y = types[2].as<TensorTypeNode>();
  CHECK(condition != nullptr && x != nullptr && y != nullptr);

  const auto& cond_shape = condition->shape;
  const auto& x_shape = x->shape;
  const auto& y_shape = y->shape;
  CHECK(x_shape.size() == y_shape.size()) << "x and y must have the same size";

  if (cond_shape.size() != x_shape.size()) {
    CHECK_EQ(cond_shape.size(), 1) << "Shape of condition " << condition->shape
                                   << " must be either equal to x or has dimension of 1.";
  }
  for (size_t i = 0; i < x_shape.size(); i++) {
    CHECK(reporter->AssertEQ(x_shape[i], y_shape[i]))
        << "x and y must have the same shape: " << x_shape << " vs " << y_shape;

    if (i < cond_shape.size()) {
      CHECK(reporter->AssertEQ(cond_shape[i], x_shape[i]))
          << "condition and x must have the same shape: " << cond_shape << " vs " << x_shape;
    }
  }
  reporter->Assign(types[3], TensorType(x_shape, x->dtype));
  return true;
}

// Positional relay function to create where operator.
Expr MakeWhere(const Expr& condition, const Expr& x, const Expr& y) {
  static const Op& op = Op::Get("where");
  return Call(op, {condition, x, y});
}

Array<te::Tensor> WhereCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                               const Type& out_type) {
  return {topi::where(inputs[0], inputs[1], inputs[2])};
}

TVM_REGISTER_GLOBAL("relay.op._make.where").set_body_typed(MakeWhere);

RELAY_REGISTER_OP("where")
    .describe(R"code(
Return the elements, either from x or y, depending on the condition.

Given three ndarrays, condition, x, and y, return an ndarray with the elements
from x or y, depending on the elements from condition are true or false.
x and y must have the same shape. If condition has the same shape as x,
each element in the output array is from x if the corresponding element
in the condition is true, and from y if false.

If condition does not have the same shape as x, it must be a 1D array whose
size is the same as x’s first dimension size. Each row of the output array
is from x’s row if the corresponding element from condition is true, and
from y’s row if false.

Note that all non-zero values are interpreted as True in condition.

Examples::

  x = [[1, 2], [3, 4]]
  y = [[5, 6], [7, 8]]
  cond = [[0, 1], [-1, 0]]
  where(cond, x, y) = [[5, 2], [3, 8]]


  cond = [1, 0]
  where(cond, x, y) = [[1, 2], [7, 8]]

)code" TVM_ADD_FILELINE)
    .add_argument("condition", "Tensor", "Condition array")
    .add_argument("x", "Tensor", "First array to be selected")
    .add_argument("y", "Tensor", "Second array to be selected")
    .set_num_inputs(3)
    .set_support_level(4)
    .add_type_rel("Where", WhereRel)
    .set_attr<FTVMCompute>("FTVMCompute", WhereCompute)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

// Squeeze
TVM_REGISTER_NODE_TYPE(SqueezeAttrs);

Expr MakeSqueeze(Expr data, Array<Integer> axis) {
  auto attrs = make_object<SqueezeAttrs>();
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("squeeze");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.squeeze").set_body_typed(MakeSqueeze);

bool SqueezeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* param = attrs.as<SqueezeAttrs>();
  CHECK(param != nullptr);
  std::vector<IndexExpr> result_shape;
  // if axes is None, squeeze all axes of dimension 1
  if (!param->axis.defined()) {
    for (const auto& e : data->shape) {
      if (!e.as<IntImmNode>()) {
        LOG(FATAL) << "axis needs to be defined for dynamic input.";
      }
      const int64_t* axis_ptr = tir::as_const_int(e);
      CHECK(axis_ptr != nullptr) << "the axes attribute must be concrete";
      if (*axis_ptr != 1) {
        result_shape.push_back(e);
      }
    }
  } else {
    // pair up original shape with a boolean which control whether it will be in the final shape.
    std::vector<std::pair<IndexExpr, bool>> original_shape;
    for (const auto& e : data->shape) {
      original_shape.push_back(std::pair<IndexExpr, bool>(e, true));
    }
    for (const auto& e : param->axis) {
      int64_t axis_val = e->value;
      if (axis_val < 0) {
        axis_val += static_cast<int64_t>(original_shape.size());
      }
      CHECK_GE(axis_val, 0);
      CHECK_LT(axis_val, original_shape.size());
      original_shape.at(axis_val).second = false;
    }
    for (const auto& p : original_shape) {
      if (p.second) {
        result_shape.push_back(p.first);
      } else {
        const int64_t* axis_ptr = tir::as_const_int(p.first);
        CHECK(axis_ptr != nullptr) << "cannot get concrete shape of input tensor";
        CHECK_EQ(*axis_ptr, 1) << "cannot squeeze axis with dimension not equal to 1";
      }
    }
  }
  reporter->Assign(types[1], TensorType(result_shape, data->dtype));
  return true;
}

Array<te::Tensor> SqueezeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {
  const SqueezeAttrs* param = attrs.as<SqueezeAttrs>();
  CHECK(param != nullptr);
  return {topi::squeeze(inputs[0], param->axis)};
}

RELAY_REGISTER_OP("squeeze")
    .describe(R"code(Squeeze the input tensor at the dimensions given by axes

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<SqueezeAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Squeeze", SqueezeRel)
    .set_attr<FTVMCompute>("FTVMCompute", SqueezeCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// CollapseSumLike: <A, B> -> B where BroadCast(A, B) = A
bool CollapseSumLikeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  reporter->Assign(types[2], types[1]);
  return BroadcastRel({types[0], types[1], types[0]}, 2, Attrs(), reporter);
}

Expr MakeCollapseSumLike(Expr data, Expr collapse_type) {
  static const Op& op = Op::Get("collapse_sum_like");
  return Call(op, {data, collapse_type}, Attrs(), {});
}

Array<te::Tensor> CollapseSumLikeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                         const Type& out_type) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  CHECK(out_ttype != nullptr);
  return {topi::collapse_sum(inputs[0], out_ttype->shape)};
}

TVM_REGISTER_GLOBAL("relay.op._make.collapse_sum_like").set_body_typed(MakeCollapseSumLike);

RELAY_REGISTER_OP("collapse_sum_like")
    .describe(R"code(Collapse the first input to match the shape of the second input.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("collapse_type", "Tensor", "Provide the type to collapse to.")
    .set_support_level(10)
    .add_type_rel("CollapseSumLike", CollapseSumLikeRel)
    .set_attr<FTVMCompute>("FTVMCompute", CollapseSumLikeCompute)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

// CollapseSumTo: <A, B> -> B where Broadcast(A, B) = A
bool CollapseSumToRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  const auto* target_shape = types[1].as<TensorTypeNode>();
  DataType out_dtype = types[0].as<TensorTypeNode>()->dtype;

  const IntImmNode* shape_shape = target_shape->shape[0].as<IntImmNode>();
  CHECK(shape_shape) << "Parameter shape must have static shape";

  std::vector<IndexExpr> oshape;
  if (param->shape) {
    const Array<Integer>& cshape_array = param->shape.value();
    for (size_t i = 0; i < cshape_array.size(); ++i) {
      oshape.push_back(cshape_array[i]);
    }
  } else {
    for (int i = 0; i < shape_shape->value; ++i) {
      oshape.push_back(Any());
    }
  }
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return BroadcastRel({types[0], types[2], types[0]}, 2, Attrs(), reporter);
}

Expr MakeCollapseSumTo(Expr data, Expr shape) {
  static const Op& op = Op::Get("collapse_sum_to");
  auto attrs = make_object<InitOpAttrs>();
  if (const auto* cshape = shape.as<ConstantNode>()) {
    attrs->shape = ToVector(cshape->data);
  }
  return Call(op, {data, shape}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.collapse_sum_to").set_body_typed(MakeCollapseSumTo);

RELAY_REGISTER_OP("collapse_sum_to")
    .describe(R"code(Broadcast the first input to match the shape argument.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Tensor", "Target shape.")
    .set_support_level(4)
    .add_type_rel("CollapseSumTo", CollapseSumToRel)
    .set_attr<FTVMCompute>("FTVMCompute", CollapseSumLikeCompute)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

// BroadCastTo: <A, B> -> B where BroadCast(A, B) = B
bool BroadCastToRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  const auto* target_shape = types[1].as<TensorTypeNode>();
  DataType out_dtype = types[0].as<TensorTypeNode>()->dtype;

  const IntImmNode* shape_shape = target_shape->shape[0].as<IntImmNode>();
  CHECK(shape_shape) << "Parameter shape must have static shape";

  std::vector<IndexExpr> oshape;
  if (param->shape) {
    const Array<Integer>& cshape_array = param->shape.value();
    for (size_t i = 0; i < cshape_array.size(); ++i) {
      oshape.push_back(cshape_array[i]);
    }
  } else {
    for (int i = 0; i < shape_shape->value; ++i) {
      oshape.push_back(Any());
    }
  }
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return BroadcastRel({types[0], types[2], types[2]}, 2, Attrs(), reporter);
}

Expr MakeBroadCastTo(Expr data, Expr shape) {
  static const Op& op = Op::Get("broadcast_to");
  auto attrs = make_object<InitOpAttrs>();
  if (const auto* cshape = shape.as<ConstantNode>()) {
    attrs->shape = ToVector(cshape->data);
  }
  return Call(op, {data, shape}, Attrs(attrs), {});
}

Array<te::Tensor> BroadCastToCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                     const Type& out_type) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  return {topi::broadcast_to(inputs[0], out_ttype->shape)};
}

TVM_REGISTER_GLOBAL("relay.op._make.broadcast_to").set_body_typed(MakeBroadCastTo);

RELAY_REGISTER_OP("broadcast_to")
    .describe(R"code(Broadcast the first input to match the shape argument.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Tensor", "Target shape.")
    .set_support_level(4)
    .add_type_rel("BroadCastTo", BroadCastToRel)
    .set_attr<FTVMCompute>("FTVMCompute", BroadCastToCompute)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

// BroadCastToLike: <A, B> -> B where BroadCast(A, B) = B
bool BroadCastToLikeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  reporter->Assign(types[2], types[1]);
  return BroadcastRel({types[0], types[1], types[1]}, 2, Attrs(), reporter);
}

Expr MakeBroadCastToLike(Expr data, Expr broadcast_type) {
  static const Op& op = Op::Get("broadcast_to_like");
  return Call(op, {data, broadcast_type}, Attrs(), {});
}

Array<te::Tensor> BroadCastToLikeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                         const Type& out_type) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  CHECK(out_ttype != nullptr);
  return {topi::broadcast_to(inputs[0], out_ttype->shape)};
}

TVM_REGISTER_GLOBAL("relay.op._make.broadcast_to_like").set_body_typed(MakeBroadCastToLike);

RELAY_REGISTER_OP("broadcast_to_like")
    .describe(R"code(Broadcast the first input to match the shape of the second input.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("broadcast_type", "Tensor", "Provide the type to broadcast to.")
    .set_support_level(10)
    .add_type_rel("BroadCastToLike", BroadCastToLikeRel)
    .set_attr<FTVMCompute>("FTVMCompute", BroadCastToLikeCompute)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

// Adapter function to make int array.
Array<Integer> GetIntArray(Array<IndexExpr> arr) {
  for (size_t i = 0; i < arr.size(); ++i) {
    CHECK(!arr[i].defined() || arr[i].as<IntImmNode>()) << "Expect an int array";
  }
  return Downcast<Array<Integer>>(arr);
}

// strided_slice
TVM_REGISTER_NODE_TYPE(StridedSliceAttrs);

bool StridedSliceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const StridedSliceAttrs* param = attrs.as<StridedSliceAttrs>();
  CHECK(param != nullptr);
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  auto dshape = data->shape;
  int64_t num_axis = dshape.size();

  // calculate output shape
  std::vector<IndexExpr> oshape(num_axis);
  if (param->begin && param->end && param->strides) {
    // stride will be set as 1 if slice mode is enabled
    std::vector<int64_t> stride_vec(num_axis, 1);
    if (param->slice_mode == "end") {
      for (size_t i = 0; i < param->strides.value().size(); ++i) {
        CHECK(param->strides.value()[i].defined());
        stride_vec[i] = param->strides.value()[i]->value;
      }
    }
    const int64_t max_range = std::numeric_limits<int64_t>::max();
    std::vector<int64_t> begin_vec;
    for (size_t i = 0; i < param->begin.value().size(); ++i) {
      if (!param->begin.value()[i].defined()) {
        begin_vec.push_back(stride_vec[i] > 0 ? 0 : max_range);
      } else {
        begin_vec.push_back(param->begin.value()[i]->value);
      }
    }
    for (int64_t i = begin_vec.size(); i < num_axis; ++i) {
      begin_vec.push_back(stride_vec[i] > 0 ? 0 : max_range);
    }

    std::vector<int64_t> end_vec;
    for (size_t i = 0; i < param->end.value().size(); ++i) {
      // allow end to be None
      if (!param->end.value()[i].defined()) {
        end_vec.push_back(stride_vec[i] < 0 ? 0 : max_range);
      } else if (param->slice_mode == "size") {
        if (param->end.value()[i]->value < 0) {
          end_vec.push_back(max_range);
        } else {
          end_vec.push_back(begin_vec[i] + param->end.value()[i]->value);
        }
      } else if (param->slice_mode == "end") {
        end_vec.push_back(param->end.value()[i]->value);
      } else {
        LOG(FATAL) << "Unsupported slice mode: " << param->slice_mode;
      }
    }
    for (int64_t i = end_vec.size(); i < num_axis; ++i) {
      end_vec.push_back(stride_vec[i] < 0 ? 0 : max_range);
    }

    for (int64_t i = 0; i < num_axis; ++i) {
      int64_t stride_v = stride_vec[i];
      int64_t begin_v = begin_vec[i];
      int64_t end_v = end_vec[i];

      if ((stride_v == 1 && begin_v == 0 && end_v == max_range) ||
          (stride_v == -1 && begin_v == max_range && end_v == 0)) {
        // Quick path, do not slice this dimension.
        oshape[i] = dshape[i];
        continue;
      }
      // Normal path, require the shape to be concrete integer.
      // Require concrete integer as symbolic inference of min/max
      // can get complicated and not very helpful.
      const int64_t* p_dim_size = tir::as_const_int(dshape[i]);
      if (!p_dim_size) {
        oshape[i] = dshape[i];
        continue;
      }
      int64_t dim_size = p_dim_size[0];
      begin_v = (begin_v < 0) ? dim_size + begin_v : begin_v;
      end_v = (end_v < 0) ? dim_size + end_v : end_v;

      int64_t slice_range, step;
      if (stride_v < 0) {
        if (end_v < -1) end_v = -1;
        CHECK_LE(end_v, begin_v) << "strided_slice get empty slice at axis " << i;
        begin_v = std::min(dim_size - 1, begin_v);
        slice_range = begin_v - end_v;
        step = -stride_v;
      } else {
        if (begin_v < 0) begin_v = 0;
        CHECK_GE(stride_v, 0);
        CHECK_LE(begin_v, end_v) << "strided_slice get invalid slice at axis " << i;
        end_v = std::min(dim_size, end_v);
        slice_range = end_v - begin_v;
        step = stride_v;
      }
      oshape[i] = tir::make_const(dshape[i].dtype(), (slice_range + step - 1) / step);
    }
  } else {
    for (int64_t i = 0; i < num_axis; ++i) {
      oshape[i] = Any();
    }
  }

  reporter->Assign(types[4], TensorType(oshape, data->dtype));
  return true;
}

Array<Array<Layout>> StridedSliceInferCorrectLayout(const Attrs& attrs,
                                                    const Array<Layout>& new_in_layouts,
                                                    const Array<Layout>& old_in_layouts,
                                                    const Array<tvm::relay::Type>& old_in_types) {
  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    CHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }

  CHECK(old_in_layouts.defined());
  CHECK_GE(old_in_layouts.size(), 1);
  CHECK(old_in_shapes.defined());
  CHECK_GE(old_in_shapes.size(), 1);

  auto layout = old_in_layouts[0];
  if (layout.defined() && new_in_layouts.defined()) {
    CHECK_GE(new_in_layouts.size(), 1);
    auto new_layout = new_in_layouts[0];
    auto shape = old_in_shapes[0];

    // NOTE: Discard "const" qualifier here.
    auto* params = const_cast<StridedSliceAttrs*>(attrs.as<StridedSliceAttrs>());
    CHECK(params != nullptr);
    Array<Integer> begin, end, strides;
    if (params->begin && params->end && params->strides) {
      for (Integer i : params->strides.value()) {
        CHECK(i.defined());
        strides.push_back(params->slice_mode == "size" ? 1 : i->value);
      }

      for (Integer i : params->begin.value()) {
        CHECK(i.defined());
        begin.push_back(i->value);
      }
      for (Integer i : params->end.value()) {
        CHECK(i.defined());
        end.push_back(i->value);
      }
    }

    Array<Integer> new_begin, new_end;

    for (size_t i = 0; i < begin.size(); i++) {
      const LayoutAxis& axis = layout[i];
      if (!axis.IsPrimal()) {
        // original layout that contains splitted axes is not supported
        return {{Layout::Undef()}, {Layout::Undef()}};
      }
      auto factor = new_layout.FactorOf(axis);
      if (factor == -1) {
        new_begin.push_back(begin[i]);
        new_end.push_back(end[i]);
      } else {
        if (strides.defined() && i < strides.size()) {
          auto stride = strides[i];
          // arbitrary stride is not supported
          if (stride.defined() && stride->value != 1) {
            return {{Layout::Undef()}, {Layout::Undef()}};
          }
        }
        int64_t bg = begin[i].defined() ? begin[i]->value : 0;
        int64_t ed;
        if (!end[i].defined()) {
          ed = shape[i].as<IntImmNode>()->value;
        } else if (params->slice_mode == "size") {
          if (end[i]->value < 0) {
            ed = shape[i].as<IntImmNode>()->value;
          } else {
            ed = bg + end[i]->value;
          }
        } else {
          ed = end[i]->value;
        }

        if (bg % factor || ed % factor) {
          // transform to original layout
          return {{Layout::Undef()}, {Layout::Undef()}};
        }
        new_begin.push_back(tvm::Integer(bg / factor));
        new_end.push_back(tvm::Integer(ed / factor));
      }
    }

    layout = new_layout;
    params->begin = new_begin;
    params->end = new_end;
  }
  return {{layout, Layout("C"), Layout("C"), Layout("C")}, {layout}};
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
  const StridedSliceAttrs* param = attrs.as<StridedSliceAttrs>();
  CHECK(param != nullptr);
  if (param->begin && param->end && param->strides) {
    Array<Integer> begin, end, strides;
    begin = param->begin.value();
    end = param->end.value();
    strides = param->strides.value();
    return Array<te::Tensor>{
        topi::strided_slice(inputs[0], begin, end, strides, param->slice_mode)};
  } else {
    te::Tensor data = inputs[0];
    te::Tensor begin = inputs[1];
    te::Tensor end = inputs[2];
    te::Tensor strides = inputs[3];
    // Dynamic computation
    int64_t attr_size = data->shape.size();
    CHECK(begin->shape[0].as<IntImmNode>()->value == attr_size &&
          end->shape[0].as<IntImmNode>()->value == attr_size &&
          strides->shape[0].as<IntImmNode>()->value == attr_size)
        << "begin, end, and strides are required to have the same length"
        << " if they are non-constant.";
    return Array<te::Tensor>{DynamicStridedSlice(data, begin, end, strides)};
  }
}

// Positional relay function to create StridedSlice operator used by frontend FFI.
Expr MakeStridedSlice(Expr data, Expr begin, Expr end, Expr strides, String slice_mode) {
  auto attrs = make_object<StridedSliceAttrs>();
  const ConstantNode *cbegin, *cend, *cstrides;
  if ((cbegin = begin.as<ConstantNode>()) && (cend = end.as<ConstantNode>()) &&
      (cstrides = strides.as<ConstantNode>())) {
    CHECK_EQ(cbegin->data->ndim, 1);
    CHECK_EQ(cend->data->ndim, 1);
    CHECK_EQ(cstrides->data->ndim, 1);
    Array<Integer> begin, end, strides;
    begin = ToVector(cbegin->data);
    end = ToVector(cend->data);
    strides = ToVector(cstrides->data);
    attrs->begin = begin;
    attrs->end = end;
    attrs->strides = strides;
  }
  attrs->slice_mode = slice_mode;
  static const Op& op = Op::Get("strided_slice");
  return Call(op, {data, begin, end, strides}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.strided_slice").set_body_typed(MakeStridedSlice);

RELAY_REGISTER_OP("strided_slice")
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
    .add_type_rel("StridedSlice", StridedSliceRel)
    .set_attr<FTVMCompute>("FTVMCompute", StridedSliceCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_attr<AnyCodegenStrategy>("AnyCodegenStrategy", kVariableDimensions)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", StridedSliceInferCorrectLayout);

// strided_set
bool StridedSetRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 6);
  reporter->Assign(types[5], types[0]);
  return true;
}

Expr MakeStridedSet(Expr data, Expr v, Expr begin, Expr end, Expr strides) {
  static const Op& op = Op::Get("strided_set");
  return Call(op, {data, v, begin, end, strides}, {});
}

TVM_REGISTER_GLOBAL("relay.op._make.strided_set").set_body_typed(MakeStridedSet);

RELAY_REGISTER_OP("strided_set")
    .describe(R"code(Strided set of an array.
Example::

  x = [[  1.,   4.,   7.,  10.],
       [  2.,   5.,   8.,  11.],
       [  3.,   6.,   9.,  12.]]

  v = [[ 11., 22., 33.]
       [ 44., 55., 66.]]

  strided_set(x, v, begin=[0, 1], end=[2, 4], stride=[1, 1]) = \
      [[  1.,  11.,  22.,  33.],
       [  2.,  44.,  55.,  66.],
       [  3.,   6.,   9.,  12.]]
)code" TVM_ADD_FILELINE)
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("v", "Tensor", "The data to set.")
    .add_argument("begin", "Tensor", "Indices for the start of the slice.")
    .add_argument("end", "Tensor", "Indices indicating the end of the slice.")
    .add_argument("strides", "Tensor", "The strides values.")
    .set_support_level(4)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .add_type_rel("StridedSet", StridedSetRel);

// relay.split
TVM_REGISTER_NODE_TYPE(SplitAttrs);

bool SplitRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
              const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  CHECK_NE(data->shape.size(), 0) << "Input shape cannot be empty";
  const auto param = attrs.as<SplitAttrs>();
  CHECK(param != nullptr);
  auto axis = param->axis;
  if (axis < 0) {
    axis += data->shape.size();
  }
  CHECK_LT(axis, data->shape.size()) << "axis should be within the input dimension range.";
  CHECK_GE(axis, 0) << "axis should be within the input dimension range.";

  if (const IntImmNode* sections = param->indices_or_sections.as<IntImmNode>()) {
    if (!data->shape[axis].as<AnyNode>()) {
      CHECK(reporter->Assert(indexmod(data->shape[axis], sections->value) ==
                             tir::make_zero(DataType::Int(64))))
          << "indices_or_sections need to be able to divide input.shape[axis]";
    }
    std::vector<Type> fields;
    for (int i = 0; i < sections->value; ++i) {
      std::vector<IndexExpr> oshape(data->shape.begin(), data->shape.end());
      if (data->shape[axis].as<AnyNode>()) {
        oshape[axis] = Any();
      } else {
        oshape[axis] = indexdiv(oshape[axis], sections->value);
      }
      auto vec_type = TensorType(oshape, data->dtype);
      fields.push_back(vec_type);
    }
    reporter->Assign(types[1], TupleType(Array<Type>(fields)));
  } else {
    auto indices = Downcast<Array<ObjectRef>>(param->indices_or_sections);
    auto begin = IndexExpr(tir::make_zero(DataType::Int(32)));
    std::vector<Type> fields;
    for (unsigned int i = 0; i < indices.size(); ++i) {
      CHECK(reporter->Assert(Downcast<IndexExpr>(indices[i]) > begin))
          << "indices_or_sections need to be a sorted ascending list";
      std::vector<IndexExpr> oshape(data->shape.begin(), data->shape.end());
      oshape[axis] = Downcast<IndexExpr>(indices[i]) - begin;
      begin = Downcast<IndexExpr>(indices[i]);
      auto vec_type = TensorType(oshape, data->dtype);
      fields.push_back(vec_type);
    }
    if (!data->shape[axis].as<AnyNode>()) {
      CHECK(reporter->Assert(begin < data->shape[axis]))
          << "The sum of sections must match the input.shape[axis]";
    }
    std::vector<IndexExpr> oshape(data->shape.begin(), data->shape.end());
    if (data->shape[axis].as<AnyNode>()) {
      oshape[axis] = Any();
    } else {
      oshape[axis] = data->shape[axis] - begin;
    }
    auto vec_type = TensorType(oshape, data->dtype);
    fields.push_back(vec_type);
    reporter->Assign(types[1], TupleType(Array<Type>(fields)));
  }
  return true;
}

Array<te::Tensor> SplitCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                               const Type& out_type) {
  const auto param = attrs.as<SplitAttrs>();
  CHECK(param != nullptr);

  if (const IntImmNode* sections = param->indices_or_sections.as<IntImmNode>()) {
    int64_t num_sections = sections->value;
    return Array<te::Tensor>{topi::split_sections(inputs[0], num_sections, param->axis)};
  } else {
    auto indices = Downcast<Array<Integer>>(param->indices_or_sections);
    return Array<te::Tensor>{topi::split(inputs[0], indices, param->axis)};
  }
}

Expr MakeSplit(Expr data, ObjectRef indices_or_sections, int axis) {
  auto attrs = make_object<SplitAttrs>();
  attrs->axis = axis;
  attrs->indices_or_sections = std::move(indices_or_sections);
  static const Op& op = Op::Get("split");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.split").set_body([](const TVMArgs& args, TVMRetValue* rv) {
  if (args.type_codes[1] == kDLInt) {
    // Note: we change it from Int(64) to Int(32) for now as
    // combine_parallel_dense will transform the graph with Int(32).
    // More invetigation is needs to check which one we should use.
    *rv =
        MakeSplit(args[0], tir::make_const(DataType::Int(32), static_cast<int>(args[1])), args[2]);
  } else {
    *rv = MakeSplit(args[0], args[1], args[2]);
  }
});

RELAY_REGISTER_OP("split")
    .describe(R"code(Splits an array along a particular axis into multiple sub-arrays.

Indices or sections to split into. Accepts an int or a tuple
If indices_or_sections is an integer, the input will be divided equally
along given axis. If such a split is not possible, an error is raised.

If indices_or_sections is a tuple of sorted integers,
the entries indicate where along axis the array is split.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<SplitAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Split", SplitRel)
    .set_attr<FTVMCompute>("FTVMCompute", SplitCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// relay.slice_like
TVM_REGISTER_NODE_TYPE(SliceLikeAttrs);

/*!
 * \brief SliceLikeRel User defined type constraint function.
 * \param num_inputs Number of input types in the args.
 * \param attrs The additional attributes of the operator.
 * \param reporter The reporter to report solution to.
 * \return False if the relation has not been resolved, it might be resolved later.
 *  True if this relation has been resolved.
 */
bool SliceLikeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }

  const auto* target = types[1].as<TensorTypeNode>();
  if (target == nullptr) {
    return false;
  }

  const auto param = attrs.as<SliceLikeAttrs>();
  CHECK(param != nullptr);

  const Array<IndexExpr>& dshape = data->shape;
  const Array<IndexExpr>& target_shape = target->shape;
  std::vector<IndexExpr> oshape(dshape.begin(), dshape.end());

  if (!param->axes.defined()) {
    for (size_t i = 0; i < dshape.size(); ++i) {
      if (i < target_shape.size()) {
        oshape[i] = target_shape[i];
        CHECK(reporter->Assert(oshape[i] <= dshape[i]))
            << "End index of axis " << i << " exceeds input shape: " << oshape[i] << " vs "
            << dshape[i];
      }
    }
  } else {
    CHECK(param->axes.size() != 0) << "Axes cannot be empty.";
    for (Integer val : param->axes) {
      int axis = val->value;
      if (axis < 0) {
        axis += dshape.size();
      }
      CHECK(axis < static_cast<int>(target_shape.size()))
          << "Axis " << axis << " exceeds dimension " << target_shape.size() << " of target_shape.";
      oshape[axis] = target_shape[axis];
      CHECK(reporter->Assert(oshape[axis] <= dshape[axis]))
          << "End index of axis " << axis << " exceeds input shape: " << oshape[axis] << " vs "
          << dshape[axis];
    }
  }

  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeSliceLike(Expr data, Expr shape_like, Array<Integer> axes) {
  auto attrs = make_object<SliceLikeAttrs>();
  attrs->axes = std::move(axes);
  static const Op& op = Op::Get("slice_like");
  return Call(op, {data, shape_like}, Attrs(attrs), {});
}

Array<te::Tensor> SliceLikeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                   const Type& out_type) {
  const auto* param = attrs.as<SliceLikeAttrs>();
  CHECK(param != nullptr);
  Array<IndexExpr> src_shape = inputs[0]->shape;
  Array<IndexExpr> target_shape = inputs[1]->shape;
  Array<IndexExpr> begin_idx, end_idx, strides;
  for (size_t i = 0; i < src_shape.size(); ++i) {
    begin_idx.push_back(0);
    strides.push_back(1);
  }
  end_idx = Array<IndexExpr>(src_shape);
  if (!param->axes.defined()) {
    for (size_t i = 0; i < src_shape.size(); ++i) {
      if (i < target_shape.size()) {
        end_idx.Set(i, target_shape[i]);
        CHECK_LE(topi::GetConstInt(end_idx[i]), topi::GetConstInt(src_shape[i]))
            << "End index of axis " << i
            << " exceeds input shape: " << topi::GetConstInt(end_idx[i]) << " vs "
            << topi::GetConstInt(src_shape[i]);
      }
    }
  } else {
    for (int axis : param->axes) {
      if (axis < 0) {
        axis = static_cast<int>(src_shape.size()) + axis;
      }
      end_idx.Set(axis, target_shape[axis]);
      CHECK_LE(topi::GetConstInt(end_idx[axis]), topi::GetConstInt(src_shape[axis]))
          << "End index of axis " << axis
          << " exceeds input shape: " << topi::GetConstInt(end_idx[axis]) << " vs "
          << topi::GetConstInt(src_shape[axis]);
    }
  }
  return Array<te::Tensor>{topi::strided_slice(inputs[0], GetIntArray(begin_idx),
                                               GetIntArray(end_idx), GetIntArray(strides), "end")};
}

TVM_REGISTER_GLOBAL("relay.op._make.slice_like").set_body_typed(MakeSliceLike);

RELAY_REGISTER_OP("slice_like")
    .describe(R"code(Slice the first input respect to the second input.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<SliceLikeAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape_like", "Tensor", "Shape tensor.")
    .set_support_level(10)
    .add_type_rel("SliceLike", SliceLikeRel)
    .set_attr<FTVMCompute>("FTVMCompute", SliceLikeCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// relay.layout_transform
TVM_REGISTER_NODE_TYPE(LayoutTransformAttrs);

Array<te::Tensor> LayoutTransformCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                         const Type& out_type) {
  const auto* param = attrs.as<LayoutTransformAttrs>();
  CHECK(param != nullptr);
  return Array<te::Tensor>{topi::layout_transform(inputs[0], param->src_layout, param->dst_layout)};
}

bool LayoutTransformRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "LayoutTransform: expect input data type to be TensorType but get " << types[0];
    return false;
  }
  const LayoutTransformAttrs* params = attrs.as<LayoutTransformAttrs>();

  Layout src_layout(params->src_layout);
  Layout dst_layout(params->dst_layout);

  CHECK(src_layout.defined() && dst_layout.defined()) << "cannot convert from/to undefined layout";
  auto layout_converter = tir::BijectiveLayout(src_layout, dst_layout);
  CHECK(layout_converter.defined())
      << "cannot convert from " << params->src_layout << " to " << params->dst_layout;

  const auto& out_shape = layout_converter.ForwardShape(data->shape);
  reporter->Assign(types[1], TensorType(out_shape, data->dtype));
  return true;
}

Expr MakeLayoutTransform(Expr data, String src_layout, String dst_layout) {
  auto attrs = make_object<LayoutTransformAttrs>();
  attrs->src_layout = std::move(src_layout);
  attrs->dst_layout = std::move(dst_layout);
  static const Op& op = Op::Get("layout_transform");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.layout_transform").set_body_typed(MakeLayoutTransform);

RELAY_REGISTER_OP("layout_transform")
    .describe(R"code(Transform the input data layout.

For transforming from NCHW to N16cHWC, the `__layout_transform__` operator reshapes
the input array by output[n, c, h, w, C] = data[n, C*16+c, h, w]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<LayoutTransformAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_type_rel("layout_transform", LayoutTransformRel)
    .set_support_level(5)
    .set_attr<FTVMCompute>("FTVMCompute", LayoutTransformCompute);

/* relay._contrib_reverse_reshape */
Expr MakeReverseReshape(Expr data, Array<Integer> newshape) {
  auto attrs = make_object<ReshapeAttrs>();
  attrs->newshape = std::move(newshape);
  attrs->reverse = true;
  static const Op& op = Op::Get("_contrib_reverse_reshape");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make._contrib_reverse_reshape").set_body_typed(MakeReverseReshape);

RELAY_REGISTER_OP("_contrib_reverse_reshape")
    .describe(R"code(Reshapes the input array where the special values are inferred from
right to left.

Example::

The special values have the same semantics as reshape. The difference is that
special values are inferred from right to left. It can be explained in the
example below::

- data.shape = (10,5,4), newshape = (-1,0), reshape results in (40,5)
- data.shape = (10,5,4), newshape = (-1,0), reverse_reshape results in (40,5)

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<ReshapeAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(10)
    .add_type_rel("Reshape", ReshapeRel)
    .set_attr<FTVMCompute>("FTVMCompute", ReshapeCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// gather operator
TVM_REGISTER_NODE_TYPE(GatherAttrs);

bool GatherRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // `types` contains: [data, indices, result]
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* indices = types[1].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "Gather: expect input data type to be TensorType but get " << types[0];
    return false;
  }
  if (indices == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
        << "Gather: expect indices type to be TensorType but get " << types[1];
    return false;
  }
  CHECK(indices->dtype.is_int()) << "indices of take must be tensor of integer";
  const auto param = attrs.as<GatherAttrs>();
  CHECK(param != nullptr);
  CHECK(param->axis.defined());

  const auto ndim_data = data->shape.size();
  const auto ndim_indices = indices->shape.size();
  int axis = param->axis->value;
  CHECK_EQ(ndim_data, ndim_indices);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, ndim_data);

  std::vector<IndexExpr> oshape;
  oshape.reserve(ndim_data);
  for (size_t i = 0; i < ndim_data; ++i) {
    if (i == (size_t)axis) {
      const int64_t* indice_shape_i = tir::as_const_int(indices->shape[i]);
      CHECK_GE(*indice_shape_i, 1);
    } else {
      CHECK(reporter->AssertEQ(indices->shape[i], data->shape[i]));
    }
    oshape.emplace_back(indices->shape[i]);
  }
  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> GatherCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  const auto* param = attrs.as<GatherAttrs>();
  return {topi::gather(inputs[0], param->axis, inputs[1])};
}

Expr MakeGather(Expr data, Integer axis, Expr indices) {
  auto attrs = make_object<GatherAttrs>();
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("gather");
  return Call(op, {data, indices}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.gather").set_body_typed(MakeGather);

RELAY_REGISTER_OP("gather")
    .describe(R"code(Gather values along given axis from given indices.

E.g. for a 3D tensor, output is computed as:

	out[i][j][k] = data[indices[i][j][k]][j][k]  # if axis == 0
	out[i][j][k] = data[i][indices[i][j][k]][k]  # if axis == 1
	out[i][j][k] = data[i][j][indices[i][j][k]]  # if axis == 2

``indices`` must have same shape as ``data``, except at dimension ``axis``
which must just be not null. Output will have same shape as ``indices``.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<GatherAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input data to the operator.")
    .add_argument("indices", "Tensor", "The indices of values to gather.")
    .set_support_level(3)
    .add_type_rel("Gather", GatherRel)
    .set_attr<FTVMCompute>("FTVMCompute", GatherCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// gather_nd operator
bool GatherNDRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  // `types` contains: [data, indices, result]
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* indices = types[1].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "GatherND: expect input data type to be TensorType but get " << types[0];
    return false;
  }
  if (indices == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
        << "GatherND: expect indices type to be TensorType but get " << types[1];
    return false;
  }
  const size_t ndim = data->shape.size();
  const IntImmNode* mdim = indices->shape[0].as<IntImmNode>();
  const size_t kdim = indices->shape.size() - 1;
  CHECK(size_t(mdim->value) <= ndim) << "GatherND: indices shape does satisfy.";

  Array<IndexExpr> oshape;
  for (size_t i = 1; i < kdim + 1; ++i) oshape.push_back(indices->shape[i]);
  for (size_t i = mdim->value; i < ndim; ++i) oshape.push_back(data->shape[i]);
  if (oshape.size() == 0) {
    oshape.push_back(tir::make_const(DataType::Int(32), 1));
  }
  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> GatherNDCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                  const Type& out_type) {
  return {topi::gather_nd(inputs[0], inputs[1])};
}

Expr MakeGatherND(Expr data, Expr indices) {
  static const Op& op = Op::Get("gather_nd");
  return Call(op, {data, indices}, {});
}

TVM_REGISTER_GLOBAL("relay.op._make.gather_nd").set_body_typed(MakeGatherND);

RELAY_REGISTER_OP("gather_nd")
    .describe(R"code(Gather elements or slices from data and store to
                 a tensor whose shape is defined by indices.

Given data with shape (X_0, X_1, ..., X_{N-1}) and indices with
shape (M, Y_0, ..., Y_{K-1}), the output will have shape
(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}), where M <= N. If M == N,
output shape will simply be (Y_0, ..., Y_{K-1}).
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("GatherND", GatherNDRel)
    .set_attr<FTVMCompute>("FTVMCompute", GatherNDCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// relay.sequence_mask
TVM_REGISTER_NODE_TYPE(SequenceMaskAttrs);

bool SequenceMaskRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  // `types` contains: [data, valid_length, result]
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* valid_length = types[1].as<TensorTypeNode>();
  CHECK(data);
  CHECK(valid_length);
  const auto param = attrs.as<SequenceMaskAttrs>();
  Array<IndexExpr> valid_length_shape;
  CHECK(param->axis == 0 || param->axis == 1);
  valid_length_shape.push_back(data->shape[1 - param->axis]);
  reporter->Assign(types[1], TensorType(valid_length_shape, valid_length->dtype));
  reporter->Assign(types[2], types[0]);
  return true;
}

Array<te::Tensor> SequenceMaskCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                      const Type& out_type) {
  const auto* param = attrs.as<SequenceMaskAttrs>();
  CHECK(param != nullptr);
  return Array<te::Tensor>{
      topi::sequence_mask(inputs[0], inputs[1], param->mask_value, param->axis)};
}

Expr MakeSequenceMask(Expr data, Expr valid_length, double mask_value, int axis) {
  auto attrs = make_object<SequenceMaskAttrs>();
  attrs->mask_value = std::move(mask_value);
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("sequence_mask");
  return Call(op, {data, valid_length}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.sequence_mask").set_body_typed(MakeSequenceMask);

RELAY_REGISTER_OP("sequence_mask")
    .describe(
        R"code(Sets all elements outside the expected length of the sequence to a constant value.

This function takes an n-dimensional input array of the form [MAX_LENGTH, batch_size, ...] or
[batch_size, MAX_LENGTH, ...] and returns an array of the same shape.

`axis` means the axis of the length dimension and can only be 0 or 1. If axis is 0,
the data must have shape [MAX_LENGTH, batch_size, ...]. Otherwise (axis=1), the data must have
shape [batch_size, MAX_LENGTH, ...].

`valid_length` gives the length of each sequence. `valid_length` should be
a 1D int array with positive ints and has dimension [batch_size,].

Examples::

  x = [[[  1.,   2.,   3.],
        [  4.,   5.,   6.]],

       [[  7.,   8.,   9.],
        [ 10.,  11.,  12.]],

       [[ 13.,  14.,   15.],
        [ 16.,  17.,   18.]]]

  // valid_length [1, 1] means only the first block of each batch will be kept
  // and other blocks are masked with default mask value = 0
  sequence_mask(x, valid_length=[1, 1]) =
       [[[  1.,   2.,   3.],
         [  4.,   5.,   6.]],

        [[  0.,   0.,   0.],
         [  0.,   0.,   0.]],

        [[  0.,   0.,   0.],
         [  0.,   0.,   0.]]]

  // valid_length [2, 3] means the first 2 blocks of the 1st batch will be kept
  // and the first 3 blocks of the 2nd batch will be kept
  // the masked values are set to be the specified mask value = 0.1
  sequence_mask(x, valid_length=[2, 3], mask_value=0.1) =
       [[[  1.,   2.,   3.],
         [  4.,   5.,   6.]],

        [[  7.,   8.,   9.],
         [  10.,  11.,  12.]],

        [[  0.1,  0.1,  0.1],
         [  16.,  17.,  18.]]]
)code" TVM_ADD_FILELINE)
    .set_attrs_type<SequenceMaskAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("valid_length", "Tensor", "The real (valid) length of each sequence.")
    .set_support_level(10)
    .add_type_rel("SequenceMask", SequenceMaskRel)
    .set_attr<FTVMCompute>("FTVMCompute", SequenceMaskCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// relay.one_hot
TVM_REGISTER_NODE_TYPE(OneHotAttrs);

bool OneHotRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // `types` contains: [indices, on_value, off_value, result]
  CHECK_EQ(types.size(), 4);
  const auto* indices = types[0].as<TensorTypeNode>();
  CHECK(indices);

  const auto param = attrs.as<OneHotAttrs>();
  CHECK_GT(param->depth, 0);

  Array<IndexExpr> oshape;
  int ndim = indices->shape.size() + 1;
  int indices_index = 0;
  int true_axis = (param->axis == -1) ? indices->shape.size() : param->axis;
  for (int i = 0; i < ndim; i++) {
    if (i == true_axis) {
      oshape.push_back(Integer(param->depth));
    } else {
      oshape.push_back(indices->shape[indices_index++]);
    }
  }

  reporter->Assign(types[3], TensorType(oshape, param->dtype));
  return true;
}

Array<te::Tensor> OneHotCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  const auto* param = attrs.as<OneHotAttrs>();
  CHECK(param != nullptr);
  return Array<te::Tensor>{
      topi::one_hot(inputs[0], inputs[1](), inputs[2](), param->depth, param->axis, param->dtype)};
}

Expr MakeOneHot(Expr indices, Expr on_value, Expr off_value, int depth, int axis, DataType dtype) {
  auto attrs = make_object<OneHotAttrs>();
  attrs->depth = std::move(depth);
  attrs->axis = axis;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("one_hot");
  return Call(op, {indices, on_value, off_value}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.one_hot").set_body_typed(MakeOneHot);

RELAY_REGISTER_OP("one_hot")
    .describe(R"code(Returns a one-hot tensor where the locations repsented by indices take value 1,
    other locations take value 0. Final dimension is <indices dimensions> x depth.

    **indices** Locations to set to 1.

    **on_value** Value to fill at indices.

    **off_value** Value to fill at all other positions besides indices.

    **depth** Depth of the one-hot dimension.

    **axis** Axis to fill.

    **dtype**)code" TVM_ADD_FILELINE)
    .set_attrs_type<OneHotAttrs>()
    .set_num_inputs(3)
    .add_argument("indices", "Tensor", "Locations to set to on_value.")
    .add_argument("on_value", "Expr", "Value to fill at indices.")
    .add_argument("off_value", "Expr", "Value to fill at all other positions besides indices.")
    .set_support_level(10)
    .add_type_rel("OneHot", OneHotRel)
    .set_attr<FTVMCompute>("FTVMCompute", OneHotCompute)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

/* relay.unravel_index */
bool UnRavelIndexRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);

  const auto* indices = types[0].as<TensorTypeNode>();
  if (indices == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "unravel_index: expect input type to be TensorType but get " << types[0];
    return false;
  }
  CHECK(indices->dtype.is_int()) << "indices of unravel_index must be tensor of integer";

  const auto* shape = types[1].as<TensorTypeNode>();
  if (shape == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
        << "unravel_index: expect input type to be TensorType but get " << types[1];
    return false;
  }
  CHECK(indices->dtype.is_int()) << "shape of unravel_index must be tensor of integer";

  Array<IndexExpr> indices_shape;
  Array<IndexExpr> shape_shape;
  indices_shape = indices->shape;
  shape_shape = shape->shape;

  Array<IndexExpr> oshape;
  oshape.push_back(shape_shape[0]);
  if (indices_shape.size() != 0) {
    oshape.push_back(indices_shape[0]);
  }
  reporter->Assign(types[2], TensorType(oshape, indices->dtype));
  return true;
}

Array<te::Tensor> UnRavelIndexCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                      const Type& out_type) {
  return Array<te::Tensor>{topi::unravel_index(inputs[0], inputs[1])};
}

Expr MakeUnRavelIndex(Expr data, Expr shape) {
  static const Op& op = Op::Get("unravel_index");
  return Call(op, {data, shape}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.unravel_index").set_body_typed(MakeUnRavelIndex);

RELAY_REGISTER_OP("unravel_index")
    .describe(
        R"code(Converts a flat index or array of flat indices into a tuple of coordinate arrays.

Example::
  -  unravel_index([22, 41, 37], (7, 6)) = [[3, 6, 6], [4, 5, 1]]
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_support_level(3)
    .add_type_rel("UnRavelIndexRel", UnRavelIndexRel)
    .set_attr<FTVMCompute>("FTVMCompute", UnRavelIndexCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// sparse_to_dense
TVM_REGISTER_NODE_TYPE(SparseToDenseAttrs);

bool SparseToDenseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(num_inputs, 3);
  auto sparse_indices = types[0].as<TensorTypeNode>();
  auto sparse_values = types[1].as<TensorTypeNode>();
  auto default_value = types[2].as<TensorTypeNode>();
  CHECK(sparse_indices != nullptr && sparse_values != nullptr && default_value != nullptr);

  CHECK(sparse_indices->dtype.is_int()) << "sparse_indices must be tensor of integers";

  CHECK_LE(sparse_indices->shape.size(), 3)
      << "sparse_indices must be a tensor of either 0D, 1D or 2D";

  CHECK_LE(sparse_values->shape.size(), 2) << "sparse_values must be a tensor of either 0D, 1D";

  CHECK_EQ(default_value->shape.size(), 0) << "default_value should be a scalar";

  const auto* param = attrs.as<SparseToDenseAttrs>();
  CHECK(param != nullptr);

  Array<IndexExpr> oshape;
  for (auto i : param->output_shape) {
    oshape.push_back(i);
  }
  reporter->Assign(types[3], TensorType(oshape, sparse_values->dtype));
  return true;
}

Array<te::Tensor> SparseToDenseCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                       const Type& out_type) {
  CHECK_EQ(inputs.size(), 3);
  const auto* param = attrs.as<SparseToDenseAttrs>();
  CHECK(param != nullptr);
  return {topi::sparse_to_dense(inputs[0], param->output_shape, inputs[1], inputs[2]())};
}

TVM_REGISTER_GLOBAL("relay.op._make.sparse_to_dense")
    .set_body_typed([](Expr indices, Array<Integer> output_shape, Expr values, Expr default_value) {
      auto attrs = make_object<SparseToDenseAttrs>();
      attrs->output_shape = std::move(output_shape);
      static const Op& op = Op::Get("sparse_to_dense");
      return Call(op, {indices, values, default_value}, Attrs(attrs));
    });

RELAY_REGISTER_OP("sparse_to_dense")
    .describe(R"code(A dense tensor from a sparse representation.

    - **sparse_indices**: A 0-D, 1-D, or 2-D tensor of integers containing location of sparse values

    - **output_shape**: A list of integers. Shape of the dense output tensor.

    - **sparse_values**: A 0-D or 1-D tensor containing the sparse values for the sparse indices.

    - **default_value**: A 0-D tensor containing the default value for the remaining locations. Defaults to 0.

    Example::
      -  sparse_to_dense([0, 0], [1, 2]], [3, 4], [1, 2], 0) = [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .set_support_level(3)
    .set_attrs_type<SparseToDenseAttrs>()
    .add_argument("sparse_indices", "Tensor", "Contains sparse indices.")
    .add_argument("sparse_values", "Tensor", "Contains values for sparse indices.")
    .add_argument("default_value", "Tensor", "Value to set for non-sparse indices. Defaults to 0.")
    .add_type_rel("SparseToDense", SparseToDenseRel)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute", SparseToDenseCompute);

}  // namespace relay
}  // namespace tvm
