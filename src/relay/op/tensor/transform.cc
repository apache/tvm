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
 * \file transform.cc
 * \brief Transform operators.
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/expr_operator.h>
#include <tvm/ir.h>
#include <tvm/data_layout.h>
#include <topi/transform.h>
#include <topi/elemwise.h>
#include <topi/broadcast.h>
#include <topi/reduction.h>
#include <topi/nn.h>
#include <vector>
#include "../op_common.h"
#include "../../../arithmetic/compute_expr.h"
#include "../../pass/alter_op_layout.h"

namespace tvm {
namespace relay {
using ir::IntImm;

// relay.cast
TVM_REGISTER_NODE_TYPE(CastAttrs);

bool CastRel(const Array<Type>& types,
             int num_inputs,
             const Attrs& attrs,
             const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "cast: expect input type to be TensorType but get "
        << types[0];
    return false;
  }
  const auto* param = attrs.as<CastAttrs>();
  reporter->Assign(types[1], TensorTypeNode::make(
      data->shape, param->dtype));
  return true;
}

Array<Tensor> CastCompute(const Attrs& attrs,
                          const Array<Tensor>& inputs,
                          const Type& out_type,
                          const Target& target) {
  const CastAttrs *param = attrs.as<CastAttrs>();
  CHECK(param != nullptr);
  DataType dtype = param->dtype;
  return { topi::cast(inputs[0], dtype) };
}

Expr MakeCast(Expr data,
              DataType dtype) {
  auto attrs = make_node<CastAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("cast");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay._make.cast")
.set_body_typed(MakeCast);

RELAY_REGISTER_OP("cast")
.describe(R"code(Cast the data into a new data type.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.CastAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Cast", CastRel)
.set_attr<FTVMCompute>("FTVMCompute", CastCompute)
.set_attr<TOpPattern>("TOpPattern", kElemWise)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

// relay.expand_dims
TVM_REGISTER_NODE_TYPE(ExpandDimsAttrs);

bool ExpandDimsRel(const Array<Type>& types,
                   int num_inputs,
                   const Attrs& attrs,
                   const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "expand_dims: expect input type to be TensorType but get "
        << types[0];
    return false;
  }
  const auto* param = attrs.as<ExpandDimsAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  const int axis = param->axis;
  const int num_newaxis = param->num_newaxis;
  CHECK(num_newaxis >= 0)
    << "expand_dims only accepts `num_newaxis >= 0`"
    << ", but got num_newaxis = " << num_newaxis;
  CHECK(-ndim - 1 <= axis && axis <= ndim)
    << "expand_dims only accepts `axis` in [-data.ndim - 1, data.ndim]"
    << ", but got axis = " << axis
    << ", and data.ndim = " << ndim;
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
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Array<Tensor> ExpandDimsCompute(const Attrs& attrs,
                                const Array<Tensor>& inputs,
                                const Type& out_type,
                                const Target& target) {
  const ExpandDimsAttrs *param = attrs.as<ExpandDimsAttrs>();
  CHECK(param != nullptr);
  return { topi::expand_dims(inputs[0], param->axis, param->num_newaxis) };
}

Expr MakeExpandDims(Expr data,
                    int axis,
                    int num_newaxis) {
  auto attrs = make_node<ExpandDimsAttrs>();
  attrs->axis = axis;
  attrs->num_newaxis = num_newaxis;
  static const Op& op = Op::Get("expand_dims");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.expand_dims")
.set_body_typed(MakeExpandDims);

RELAY_REGISTER_OP("expand_dims")
.describe(R"code(Insert `num_newaxis` axises at the position given by `axis`

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.ExpandDimsAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(1)
.add_type_rel("ExpandDims", ExpandDimsRel)
.set_attr<FTVMCompute>("FTVMCompute", ExpandDimsCompute)
.set_attr<TOpPattern>("TOpPattern", kBroadcast);

// relay.concatenate
TVM_REGISTER_NODE_TYPE(ConcatenateAttrs);

bool ConcatenateRel(const Array<Type>& types,
                    int num_inputs,
                    const Attrs& attrs,
                    const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "cast: expect input type to be TupleType but get "
        << types[0];
    return false;
  }
  const auto* param = attrs.as<ConcatenateAttrs>();
  const auto& first = Downcast<TensorType>(tensor_tuple->fields[0]);
  // Sanity check: ndim and dtype.
  const int ndim = static_cast<int>(first->shape.size());
  const DataType dtype = first->dtype;
  for (const Type& ele : tensor_tuple->fields) {
    const auto& e = Downcast<TensorType>(ele);
    int e_ndim = static_cast<int>(e->shape.size());
    const DataType& e_dtype = e->dtype;
    CHECK_EQ(e_ndim, ndim) << "relay.concatenate requires all tensors have the same ndim";
    CHECK_EQ(e_dtype, dtype) << "relay.concatenate requires all tensors have the same dtype";
  }
  // Sanity check: axis
  int axis = param->axis;
  CHECK(-ndim <= axis && axis < ndim)
    << "concatenate only accepts `axis` in [-ndim, ndim)"
    << ", but got axis = " << axis
    << ", and ndim = " << ndim;
  axis = axis < 0 ? ndim + axis : axis;
  // Calculate shape
  std::vector<IndexExpr>&& oshape = AsVector(first->shape);
  IndexExpr &concat_dim = oshape[axis];
  for (int i = 1; i < static_cast<int>(tensor_tuple->fields.size()); ++i) {
    const auto& e = Downcast<TensorType>(tensor_tuple->fields[i]);
    concat_dim += e->shape[axis];
  }
  reporter->Assign(types[1], TensorTypeNode::make(oshape, dtype));
  return true;
}

Array<Tensor> ConcatenateCompute(const Attrs& attrs,
                          const Array<Tensor>& inputs,
                          const Type& out_type,
                          const Target& target) {
  const ConcatenateAttrs *param = attrs.as<ConcatenateAttrs>();
  CHECK(param != nullptr);
  return { topi::concatenate(inputs, param->axis) };
}

Array<Array<Layout>> ConcatenateLayout(
    const Attrs& attrs,
    const Array<Layout>& new_in_layouts,
    const Array<Layout>& old_in_layouts,
    const Array<Array<IndexExpr>> &old_in_shapes) {
  const ConcatenateAttrs* param = attrs.as<ConcatenateAttrs>();

  size_t axis = param->axis < 0 ? param->axis + old_in_shapes[0].size() :
                static_cast<size_t>(param->axis);

  Layout ret;
  if (new_in_layouts.defined()) {  // this function is called after some operators are alternated.
    const auto& concate_dim = old_in_layouts[0][axis];
    for (size_t i = 0; i < new_in_layouts.size(); ++i) {
      if (new_in_layouts[i].ndim() > axis &&
          new_in_layouts[i][axis] == concate_dim) {
        ret = new_in_layouts[i];
        break;
      }
    }
  } else {  // this function is called on the original correct relay ir
    for (size_t i = 0; i < old_in_layouts.size(); ++i) {
      if (old_in_layouts[i].defined()) {
        ret = old_in_layouts[i];
        break;
      }
    }

    if (ret.ndim() <= axis || !ret[axis].IsPrimal()) {
      return Array<Array<Layout> > {{Layout::Undef()}, {Layout::Undef()}};
    }
  }

  return Array<Array<Layout> > {Array<Layout>(old_in_layouts.size(), ret), {ret}};
}

Expr MakeConcatenate(Expr data,
                     int axis) {
  auto attrs = make_node<ConcatenateAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("concatenate");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.concatenate")
.set_body_typed(MakeConcatenate);

RELAY_REGISTER_OP("concatenate")
.describe(R"code(Concatenate the input tensors along the given axis.

- **data** : A list of tensors.

- **axis** : The axis along which the tensors are concatenated.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.ConcatenateAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input list of tensors.")
.set_support_level(1)
.add_type_rel("Concatenate", ConcatenateRel)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConcatenateLayout)
.set_attr<FTVMCompute>("FTVMCompute", ConcatenateCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);

TVM_REGISTER_NODE_TYPE(StackAttrs);

bool StackRel(const Array<Type>& types,
              int num_inputs,
              const Attrs& attrs,
              const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "cast: expect input type to be TupleType but get "
        << types[0];
    return false;
  }
  const auto* param = attrs.as<StackAttrs>();
  const auto& first = Downcast<TensorType>(tensor_tuple->fields[0]);
  // Sanity check: ndim and dtype.
  const int ndim = static_cast<int>(first->shape.size());
  const DataType dtype = first->dtype;
  for (const Type& ele : tensor_tuple->fields) {
    const auto& e = Downcast<TensorType>(ele);
    int e_ndim = static_cast<int>(e->shape.size());
    const DataType& e_dtype = e->dtype;
    CHECK_EQ(e_ndim, ndim) << "relay.stack requires all tensors have the same ndim";
    CHECK_EQ(e_dtype, dtype) << "relay.stack requires all tensors have the same dtype";
  }
  // Sanity check: axis
  int axis = param->axis;
  CHECK(-ndim <= axis && axis < ndim)
    << "stack only accepts `axis` in [-ndim, ndim)"
    << ", but got axis = " << axis
    << ", and ndim = " << ndim;
  axis = axis < 0 ? ndim + axis + 1 : axis;
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
  reporter->Assign(types[1], TensorTypeNode::make(oshape, dtype));
  return true;
}

Array<Tensor> StackCompute(const Attrs& attrs,
                           const Array<Tensor>& inputs,
                           const Type& out_type,
                           const Target& target) {
  const StackAttrs *param = attrs.as<StackAttrs>();
  CHECK(param != nullptr);
  return { topi::stack(inputs, param->axis) };
}

Expr MakeStack(Expr data,
               int axis) {
  auto attrs = make_node<StackAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("stack");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.stack")
.set_body_typed(MakeStack);

RELAY_REGISTER_OP("stack")
.describe(R"code(Stack the input tensors along the given axis.

- **data** : A list of tensors.

- **axis** : The axis along which the tensors are stacked.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.StackAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input list of tensors.")
.set_support_level(3)
.add_type_rel("Stack", StackRel)
.set_attr<FTVMCompute>("FTVMCompute", StackCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);

/* relay.transpose */
TVM_REGISTER_NODE_TYPE(TransposeAttrs);

bool TransposeRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "transpose: expect input type to be TensorType but get "
        << types[0];
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
        << ", but got axis = " << axis
        << ", and data.ndim = " << ndim;
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
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Array<Tensor> TransposeCompute(const Attrs& attrs,
                               const Array<Tensor>& inputs,
                               const Type& out_type,
                               const Target& target) {
  const auto* param = attrs.as<TransposeAttrs>();
  CHECK(param != nullptr);
  return Array<Tensor>{ topi::transpose(inputs[0], param->axes) };
}

Expr MakeTranspose(Expr data,
                   Array<Integer> axes) {
  auto attrs = make_node<TransposeAttrs>();
  attrs->axes = std::move(axes);
  static const Op& op = Op::Get("transpose");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.transpose")
.set_body_typed(MakeTranspose);

RELAY_REGISTER_OP("transpose")
.describe(R"code(Permutes the dimensions of an array.

- **data**: The input data to the operator.

- **axes**: The target axes order, reverse order if not specified.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.TransposeAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Transpose", TransposeRel)
.set_attr<FTVMCompute>("FTVMCompute", TransposeCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);

/* relay.reshape */
TVM_REGISTER_NODE_TYPE(ReshapeAttrs);

bool ReshapeRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "reshape: expect input type to be TensorType but get "
        << types[0];
    return false;
  }

  const auto* param = attrs.as<ReshapeAttrs>();
  Array<IndexExpr> data_shape;
  Array<Integer> newshape;
  if (param->reverse) {
    data_shape.assign(data->shape.rbegin(), data->shape.rend());
    newshape.assign(param->newshape.rbegin(), param->newshape.rend());
  } else {
    data_shape = data->shape;
    newshape = param->newshape;
  }
  Array<IndexExpr> oshape;
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
      oshape.push_back(data_shape[src_idx++]);
    } else if (svalue == -1) {
      // inference based on rest
      CHECK_LT(infer_idx, 0)
          << "One and only one dim can be inferred";
      infer_idx = i;
      oshape.push_back(1);
      ++src_idx;
    } else if (svalue == -2) {
      // copy all remaining dims from source
      while (src_idx < data_shape.size()) {
        oshape.push_back(data_shape[src_idx++]);
      }
    } else if (svalue == -3) {
      // merge two dims from source
      CHECK_LT(src_idx + 1, data_shape.size());
      IndexExpr d1 = data_shape[src_idx++];
      IndexExpr d2 = data_shape[src_idx++];
      oshape.push_back(d1 * d2);
    } else if (svalue == -4) {
      // split the source dim s into two dims
      // read the left dim and then the right dim (either can be -1)
      CHECK_LT(i + 2, newshape.size());
      CHECK_LT(src_idx, data_shape.size());
      IndexExpr d0 = data_shape[src_idx++];
      Integer d1 = newshape[++i];
      Integer d2 = newshape[++i];
      if (d1->value == -1) {
        CHECK(d2->value != -1)
            << "Split dims cannot both be -1.";
        oshape.push_back(d0 / d2);
        oshape.push_back(d2);
      } else {
        oshape.push_back(d1);
        if (d2->value == -1) {
          oshape.push_back(d0 / d1);
        } else {
          oshape.push_back(d2);
        }
      }
    }
  }

  if (infer_idx >= 0) {
    IndexExpr new_size = arith::ComputeReduce<tvm::ir::Mul>(oshape, 1);
    IndexExpr old_size = arith::ComputeReduce<tvm::ir::Mul>(data_shape, 1);
    oshape.Set(infer_idx, old_size / new_size);
  }

  if (param->reverse) {
    reporter->Assign(types[1], TensorTypeNode::make(
        Array<IndexExpr>(oshape.rbegin(), oshape.rend()), data->dtype));
  } else {
    reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  }
  return true;
}

Array<Tensor> ReshapeCompute(const Attrs& attrs,
                             const Array<Tensor>& inputs,
                             const Type& out_type,
                             const Target& target) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  CHECK(out_ttype != nullptr);
  return { topi::reshape(inputs[0], out_ttype->shape) };
}

Expr MakeReshape(Expr data,
                 Array<Integer> newshape) {
  auto attrs = make_node<ReshapeAttrs>();
  attrs->newshape = std::move(newshape);
  attrs->reverse = false;
  static const Op& op = Op::Get("reshape");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.reshape")
.set_body_typed(MakeReshape);

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
.set_attrs_type_key("relay.attrs.ReshapeAttrs")
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
bool ReshapeLikeRel(const Array<Type>& types,
                    int num_inputs,
                    const Attrs& attrs,
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
  CHECK(reporter->AssertEQ(data->Size(), reshape_like->Size()))
    << "Reshape inputs size should be compatible.";
  reporter->Assign(types[2], TensorTypeNode::make(reshape_like->shape, data->dtype));
  return true;
}


Expr MakeReshapeLike(Expr data,
                     Expr shape_like) {
  static const Op& op = Op::Get("reshape_like");
  return CallNode::make(op, {data, shape_like}, Attrs(), {});
}


TVM_REGISTER_API("relay.op._make.reshape_like")
.set_body_typed(MakeReshapeLike);


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


// Take
TVM_REGISTER_NODE_TYPE(TakeAttrs);

bool TakeRel(const Array<Type>& types,
             int num_inputs,
             const Attrs& attrs,
             const TypeReporter& reporter) {
  // `types` contains: [data, indices, result]
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  const auto* indices = types[1].as<TensorTypeNode>();
  CHECK(indices != nullptr);
  const auto param = attrs.as<TakeAttrs>();
  CHECK(param != nullptr);

  if (!param->axis.defined()) {
    std::vector<IndexExpr>&& oshape = AsVector(indices->shape);
    reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
    return true;
  }

  std::vector<IndexExpr> oshape;
  const auto ndim_data = static_cast<int>(data->shape.size());
  const auto ndim_indices = static_cast<int>(indices->shape.size());
  int axis = static_cast<int>(param->axis->value);
  if (axis < 0) axis += ndim_data;
  CHECK_LE(axis, ndim_data)
    << "axis should be with in data shape"
    << ", but got = " << axis;

  oshape.reserve(ndim_data - 1 + ndim_indices);
  for (int i = 0; i < axis; ++i) {
    oshape.emplace_back(data->shape[i]);
  }
  for (int i = 0; i < ndim_indices; ++i) {
    oshape.emplace_back(indices->shape[i]);
  }
  for (int i = axis+1; i < ndim_data; ++i) {
    oshape.emplace_back(data->shape[i]);
  }

  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Array<Tensor> TakeCompute(const Attrs& attrs,
                          const Array<Tensor>& inputs,
                          const Type& out_type,
                          const Target& target) {
  const auto* param = attrs.as<TakeAttrs>();
  CHECK(param != nullptr);
  if (!param->axis.defined()) {
    return Array<Tensor>{ topi::take(inputs[0], inputs[1], param->mode) };
  } else {
    return Array<Tensor>{ topi::take(inputs[0], inputs[1], param->axis, param->mode) };
  }
}

Expr MakeTake(Expr data,
              Expr indices,
              Integer axis,
              std::string mode) {
  auto attrs = make_node<TakeAttrs>();
  attrs->axis = std::move(axis);
  attrs->mode = std::move(mode);
  static const Op& op = Op::Get("take");
  return CallNode::make(op, {data, indices}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.take")
.set_body_typed(MakeTake);

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
.set_attrs_type_key("relay.attrs.TakeAttrs")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("indices", "Tensor", "The indices tensor.")
.set_support_level(2)
.add_type_rel("Take", TakeRel)
.set_attr<FTVMCompute>("FTVMCompute", TakeCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);


// Init ops
TVM_REGISTER_NODE_TYPE(InitOpAttrs);

bool FullRel(const Array<Type>& types,
             int num_inputs,
             const Attrs& attrs,
             const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  const auto* fill_value = types[0].as<TensorTypeNode>();
  if (fill_value == nullptr) {
    return false;
  }

  DataType out_dtype = param->dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = fill_value->dtype;
  }

  CHECK_EQ(fill_value->shape.size(), 0)
    << "Fill value should be a scalar but has dimension "
    << fill_value->shape.size() << ".";

  reporter->Assign(types[1], TensorTypeNode::make(param->shape, out_dtype));
  return true;
}

Array<Tensor> FullCompute(const Attrs& attrs,
                          const Array<Tensor>& inputs,
                          const Type& out_type,
                          const Target& target) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  return { topi::full(out_ttype->shape, out_ttype->dtype, inputs[0]()) };
}

Expr MakeFull(Expr fill_value,
              Array<IndexExpr> shape,
              DataType dtype) {
  auto attrs = make_node<InitOpAttrs>();
  attrs->shape = std::move(shape);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("full");
  return CallNode::make(op, {fill_value}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.full")
.set_body_typed(MakeFull);

RELAY_REGISTER_OP("full")
.describe(R"code(Fill array with scalar value.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.InitOpAttrs")
.set_num_inputs(1)
.add_argument("fill_value", "double", "The value to fill.")
.set_support_level(3)
.add_type_rel("Full", FullRel)
.set_attr<FTVMCompute>("FTVMCompute", FullCompute)
.set_attr<TOpPattern>("TOpPattern", kElemWise);

bool InitOpRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 1);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();

  reporter->Assign(types[0], TensorTypeNode::make(param->shape, param->dtype));
  return true;
}

Expr MakeZeros(Array<IndexExpr> shape,
               DataType dtype) {
  auto attrs = make_node<InitOpAttrs>();
  attrs->shape = std::move(shape);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("zeros");
  return CallNode::make(op, {}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.zeros")
.set_body_typed(MakeZeros);

RELAY_REGISTER_OP("zeros")
.describe(R"code(Fill array with zeros.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.InitOpAttrs")
.set_num_inputs(0)
.set_support_level(3)
.add_type_rel("InitOp", InitOpRel);

Expr MakeOnes(Array<IndexExpr> shape,
              DataType dtype) {
  auto attrs = make_node<InitOpAttrs>();
  attrs->shape = std::move(shape);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("ones");
  return CallNode::make(op, {}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.ones")
.set_body_typed(MakeOnes);

RELAY_REGISTER_OP("ones")
.describe(R"code(Fill array with ones.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.InitOpAttrs")
.set_num_inputs(0)
.set_support_level(3)
.add_type_rel("InitOp", InitOpRel);

bool FullLikeRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
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
    << "The fill value should be a scalar but here it has dimension "
    << fill_value->shape.size() << ".";

  reporter->Assign(types[2], TensorTypeNode::make(data->shape, data->dtype));
  return true;
}

Array<Tensor> FullLikeCompute(const Attrs& attrs,
                              const Array<Tensor>& inputs,
                              const Type& out_type,
                              const Target& target) {
  return { topi::full_like(inputs[0], inputs[1]()) };
}

Expr MakeFullLike(Expr data,
                  Expr fill_value) {
  static const Op& op = Op::Get("full_like");
  return CallNode::make(op, {data, fill_value}, Attrs(), {});
}

TVM_REGISTER_API("relay.op._make.full_like")
.set_body_typed(MakeFullLike);

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

bool ArangeRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 1);
  const ArangeAttrs* param = attrs.as<ArangeAttrs>();
  IndexExpr num_elem = tvm::cast(tvm::Int(32), tvm::ceil(
      tvm::cast(tvm::Float(32), param->stop - param->start) / param->step));
  if (const tvm::ir::IntImm* val = num_elem.as<tvm::ir::IntImm>()) {
    CHECK_GT(val->value, 0)
        << "Invalid arange attributes (start, stop, step): " << param->start
        << ", " << param->stop << ", " << param->step;
  }
  reporter->Assign(types[0], TensorTypeNode::make({num_elem}, param->dtype));
  return true;
}

Array<Tensor> ArangeCompute(const Attrs& attrs,
                            const Array<Tensor>& inputs,
                            const Type& out_type,
                            const Target& target) {
  const ArangeAttrs* param = attrs.as<ArangeAttrs>();
  return { topi::arange(param->start, param->stop, param->step, param->dtype) };
}

Expr MakeArange(tvm::Expr start,
                tvm::Expr stop,
                tvm::Expr step,
                DataType dtype) {
  auto attrs = make_node<ArangeAttrs>();
  attrs->start = std::move(start);
  attrs->stop = std::move(stop);
  attrs->step = std::move(step);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("arange");
  return CallNode::make(op, {}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.arange")
.set_body_typed(MakeArange);

RELAY_REGISTER_OP("arange")
.describe(R"code(Returns evenly spaced values within a given interval.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.ArangeAttrs")
.set_num_inputs(0)
.set_support_level(3)
.add_type_rel("Arange", ArangeRel)
.set_attr<FTVMCompute>("FTVMCompute", ArangeCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);

// repeat operator
TVM_REGISTER_NODE_TYPE(RepeatAttrs);

bool RepeatRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "repeat: expect input type to be TensorType but get "
        << types[0];
    return false;
  }
  const auto* param = attrs.as<RepeatAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  const int repeats = param->repeats;
  const int axis = param->axis;
  CHECK(repeats >= 1)
    << "repeat only accepts `repeats >= 1`"
    << ", but got repeats = " << repeats;
  CHECK(-ndim - 1 <= axis && axis <= ndim)
    << "repeat only accepts `axis` in [-data.ndim - 1, data.ndim]"
    << ", but got axis = " << axis
    << ", and data.ndim = " << ndim;
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
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Array<Tensor> RepeatCompute(const Attrs& attrs,
                            const Array<Tensor>& inputs,
                            const Type& out_type,
                            const Target& target) {
  const RepeatAttrs *param = attrs.as<RepeatAttrs>();
  CHECK(param != nullptr);
  return { topi::repeat(inputs[0], param->repeats, param->axis) };
}

Expr MakeRepeat(Expr data,
                int repeats,
                int axis) {
  auto attrs = make_node<RepeatAttrs>();
  attrs->repeats = repeats;
  attrs->axis = axis;
  static const Op& op = Op::Get("repeat");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.repeat")
.set_body_typed(MakeRepeat);

RELAY_REGISTER_OP("repeat")
.describe(R"code(Repeat elements of an array `repeats` times along axis `axis`

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.Repeat")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Repeat", RepeatRel)
.set_attr<FTVMCompute>("FTVMCompute", RepeatCompute)
.set_attr<TOpPattern>("TOpPattern", kBroadcast);

// tile operator
TVM_REGISTER_NODE_TYPE(TileAttrs);

bool TileRel(const Array<Type>& types,
             int num_inputs,
             const Attrs& attrs,
             const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "tile: expect input type to be TensorType but get "
        << types[0];
    return false;
  }
  const auto* param = attrs.as<TileAttrs>();
  const size_t ndim = data->shape.size();
  const Array<Integer>& reps = param->reps;
  // check dimension match
  CHECK(reps.defined())
    << "repetition array is not defined. data.ndim = " << ndim;
  const size_t rndim = reps.size();
  for (size_t i = 0; i < rndim; ++i) {
    if (const tvm::ir::IntImm* val = reps[i].as<tvm::ir::IntImm>()) {
      CHECK_GT(val->value, 0)
          << "Tile reps value should always be larger than 0, but get: " << val->value;
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
    for (size_t i = 0; i < ndim; ++i)
        data_shape.emplace_back(data->shape[i]);
    for (size_t i = 0; i < (ndim - rndim); ++i)
        reps_shape.emplace_back(1);
    for (size_t i = 0; i < rndim; ++i)
        reps_shape.emplace_back(reps[i]);
  } else {
    for (size_t i = 0; i < rndim; ++i)
        reps_shape.emplace_back(reps[i]);
    for (size_t i = 0; i < (rndim - ndim); ++i)
        data_shape.emplace_back(1);
    for (size_t i = 0; i < ndim; ++i)
        data_shape.emplace_back(data->shape[i]);
  }
  std::vector<IndexExpr> oshape;
  oshape.reserve(tndim);
  for (size_t i = 0; i < tndim; ++i) {
    oshape.emplace_back(data_shape[i] * reps_shape[i]);
  }
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Array<Tensor> TileCompute(const Attrs& attrs,
                          const Array<Tensor>& inputs,
                          const Type& out_type,
                          const Target& target) {
  const TileAttrs *param = attrs.as<TileAttrs>();
  CHECK(param != nullptr);
  return { topi::tile(inputs[0], param->reps) };
}

Expr MakeTile(Expr data,
              Array<Integer> reps) {
  auto attrs = make_node<TileAttrs>();
  attrs->reps = reps;
  static const Op& op = Op::Get("tile");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.tile")
.set_body_typed(MakeTile);

RELAY_REGISTER_OP("tile")
.describe(R"code(Repeat the whole array multiple times.

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.Tile")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Tile", TileRel)
.set_attr<FTVMCompute>("FTVMCompute", TileCompute)
.set_attr<TOpPattern>("TOpPattern", kBroadcast);

// reverse operator
TVM_REGISTER_NODE_TYPE(ReverseAttrs);

bool ReverseRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "reverse: expect input type to be TensorType but get "
        << types[0];
    return false;
  }
  const auto* param = attrs.as<ReverseAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  const int axis = param->axis;
  CHECK(-ndim <= axis && axis < ndim)
    << "reverse only accepts `axis` in [-data.ndim, data.ndim - 1]"
    << ", but got axis = " << axis
    << ", and data.ndim = " << ndim;
  reporter->Assign(types[1], types[0]);
  return true;
}

Array<Tensor> ReverseCompute(const Attrs& attrs,
                             const Array<Tensor>& inputs,
                             const Type& out_type,
                             const Target& target) {
  const ReverseAttrs *param = attrs.as<ReverseAttrs>();
  CHECK(param != nullptr);
  return { topi::flip(inputs[0], param->axis) };
}

Expr MakeReverse(Expr data,
                 int axis) {
  auto attrs = make_node<ReverseAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("reverse");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.reverse")
.set_body_typed(MakeReverse);

RELAY_REGISTER_OP("reverse")
.describe(R"code(Reverses the order of elements along given `axis` while preserving array shape.

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.Reverse")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Reverse", ReverseRel)
.set_attr<FTVMCompute>("FTVMCompute", ReverseCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);

// where operator
bool WhereRel(const Array<Type>& types,
              int num_inputs,
              const Attrs& attrs,
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
    CHECK_EQ(cond_shape.size(), 1)
        << "Shape of condition " << condition->shape
        << " must be either equal to x or has dimension of 1.";
  }
  for (size_t i = 0; i < x_shape.size(); i++) {
    CHECK(reporter->AssertEQ(x_shape[i], y_shape[i]))
        << "x and y must have the same shape: " << x_shape << " vs " << y_shape;

    CHECK(reporter->AssertEQ(cond_shape[i], x_shape[i]))
        << "Shape of condition " << condition->shape
        << " must be either equal to x or has dimension of 1.";
  }
  reporter->Assign(types[3], TensorTypeNode::make(x_shape, x->dtype));
  return true;
}

// Positional relay function to create where operator.
Expr MakeWhere(const Expr& condition, const Expr& x, const Expr& y) {
  static const Op& op = Op::Get("where");
  return CallNode::make(op, {condition, x, y});
}

Array<Tensor> WhereCompute(const Attrs& attrs,
                           const Array<Tensor>& inputs,
                           const Type& out_type,
                           const Target& target) {
  return { topi::where(inputs[0], inputs[1], inputs[2]) };
}

TVM_REGISTER_API("relay.op._make.where")
.set_body_typed(MakeWhere);

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

Expr MakeSqueeze(Expr data,
                 Array<Integer> axis) {
  auto attrs = make_node<SqueezeAttrs>();
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("squeeze");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.squeeze")
.set_body_typed(MakeSqueeze);


bool SqueezeRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
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
      const int64_t* axis_ptr = as_const_int(e);
      CHECK(axis_ptr != nullptr) << "the axes attribute must be concrete";
      if (*axis_ptr != 1) {
        result_shape.push_back(e);
      }
    }
  } else {
    // pair up original shape with a boolean which control whether it will be in the final shape.
    std::vector<std::pair<IndexExpr, bool> > original_shape;
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
    for (const auto p : original_shape) {
      if (p.second) {
        result_shape.push_back(p.first);
      } else {
        const int64_t* axis_ptr = as_const_int(p.first);
        CHECK(axis_ptr != nullptr) << "cannot get concrete shape of input tensor";
        CHECK_EQ(*axis_ptr, 1) << "cannot squeeze axis with dimension not equal to 1";
      }
    }
  }
  reporter->Assign(types[1], TensorTypeNode::make(result_shape, data->dtype));
  return true;
}

Array<Tensor> SqueezeCompute(const Attrs& attrs,
                             const Array<Tensor>& inputs,
                             const Type& out_type,
                             const Target& target) {
  const SqueezeAttrs *param = attrs.as<SqueezeAttrs>();
  CHECK(param != nullptr);
  return { topi::squeeze(inputs[0], param->axis) };
}


RELAY_REGISTER_OP("squeeze")
.describe(R"code(Squeeze the input tensor at the dimensions given by axes

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.SqueezeAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Squeeze", SqueezeRel)
.set_attr<FTVMCompute>("FTVMCompute", SqueezeCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);


// Have no idea how to assert the constraint.
// CollapseSumLike: <A, B> -> B where BroadCast(A, B) = A
bool CollapseSumLikeRel(const Array<Type>& types,
                        int num_inputs,
                        const Attrs& attrs,
                        const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  reporter->Assign(types[2], types[1]);
  return true;
}

Expr MakeCollapseSumLike(Expr data,
                         Expr collapse_type) {
  static const Op& op = Op::Get("collapse_sum_like");
  return CallNode::make(op, {data, collapse_type}, Attrs(), {});
}

Array<Tensor> CollapseSumLikeCompute(const Attrs& attrs,
                                     const Array<Tensor>& inputs,
                                     const Type& out_type,
                                     const Target& target) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  CHECK(out_ttype != nullptr);
  return { topi::collapse_sum(inputs[0], out_ttype->shape) };
}

TVM_REGISTER_API("relay.op._make.collapse_sum_like")
.set_body_typed(MakeCollapseSumLike);

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

// BroadCastTo: <A, B> -> B where BroadCast(A, B) = B
bool BroadCastToRel(const Array<Type>& types,
                    int num_inputs,
                    const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  auto ioattrs = attrs.as<InitOpAttrs>();
  CHECK(ioattrs);
  auto intt = types[0].as<TensorTypeNode>();
  if (intt == nullptr) { return false; }
  auto type = TensorTypeNode::make(ioattrs->shape, intt->dtype);
  reporter->Assign(types[1], type);
  return true;
}

Expr MakeBroadCastTo(Expr data, Array<IndexExpr> shape) {
  static const Op& op = Op::Get("broadcast_to");
  auto attrs = make_node<InitOpAttrs>();
  attrs->shape = std::move(shape);
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

Array<Tensor> BroadCastToCompute(const Attrs& attrs,
                                 const Array<Tensor>& inputs,
                                 const Type& out_type,
                                 const Target& target) {
  auto ioattrs = attrs.as<InitOpAttrs>();
  CHECK(ioattrs != nullptr);
  return { topi::broadcast_to(inputs[0], ioattrs->shape) };
}

TVM_REGISTER_API("relay.op._make.broadcast_to")
.set_body_typed(MakeBroadCastTo);

RELAY_REGISTER_OP("broadcast_to")
.describe(R"code(Broadcast the first input to match the shape argument.
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(4)
.add_type_rel("BroadCastTo", BroadCastToRel)
.set_attr<FTVMCompute>("FTVMCompute", BroadCastToCompute)
.set_attr<TOpPattern>("TOpPattern", kBroadcast);

// BroadCastToLike: <A, B> -> B where BroadCast(A, B) = B
bool BroadCastToLikeRel(const Array<Type>& types,
                        int num_inputs,
                        const Attrs& attrs,
                        const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  reporter->Assign(types[2], types[1]);
  return true;
}

Expr MakeBroadCastToLike(Expr data,
                         Expr broadcast_type) {
  static const Op& op = Op::Get("broadcast_to_like");
  return CallNode::make(op, {data, broadcast_type}, Attrs(), {});
}

Array<Tensor> BroadCastToLikeCompute(const Attrs& attrs,
                                     const Array<Tensor>& inputs,
                                     const Type& out_type,
                                     const Target& target) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  CHECK(out_ttype != nullptr);
  return { topi::broadcast_to(inputs[0], out_ttype->shape) };
}

TVM_REGISTER_API("relay.op._make.broadcast_to_like")
.set_body_typed(MakeBroadCastToLike);

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
    CHECK(!arr[i].defined() || arr[i].as<IntImm>())
      << "Expect an int array";
  }
  return Array<Integer>(arr.node_);
}


// strided_slice
TVM_REGISTER_NODE_TYPE(StridedSliceAttrs);
bool StridedSliceRel(const Array<Type>& types,
                     int num_inputs,
                     const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const StridedSliceAttrs *param = attrs.as<StridedSliceAttrs>();
  CHECK(param != nullptr);

  auto dshape = data->shape;
  auto num_axis = dshape.size();

  std::vector<int64_t> stride_vec;
  for (Integer i : param->strides) {
    CHECK(i.defined());
    stride_vec.push_back(i->value);
  }
  for (size_t i = stride_vec.size(); i < num_axis; ++i) {
    stride_vec.push_back(1);
  }
  const int64_t max_range = std::numeric_limits<int64_t>::max();

  std::vector<int64_t> begin_vec;
  for (size_t i = 0; i < param->begin.size(); ++i) {
    if (!param->begin[i].defined()) {
      // value=None
      begin_vec.push_back(stride_vec[i] > 0 ? 0 : max_range);
    } else {
      begin_vec.push_back(param->begin[i]->value);
    }
  }
  for (size_t i = begin_vec.size(); i < num_axis; ++i) {
    begin_vec.push_back(stride_vec[i] > 0 ? 0 : max_range);
  }

  std::vector<int64_t> end_vec;
  for (size_t i = 0; i < param->end.size(); ++i) {
    // allow end to be None
    if (!param->end[i].defined()) {
      end_vec.push_back(stride_vec[i] < 0 ? 0 : max_range);
    } else {
      end_vec.push_back(param->end[i]->value);
    }
  }
  for (size_t i = end_vec.size(); i < num_axis; ++i) {
    end_vec.push_back(stride_vec[i] < 0 ? 0 : max_range);
  }

  std::vector<IndexExpr> oshape(dshape.size());
  for (size_t i = 0; i < num_axis; ++i) {
    int64_t stride_v = stride_vec[i];
    int64_t begin_v = begin_vec[i];
    int64_t end_v = end_vec[i];

    if ((stride_v == 1 &&
         begin_v == 0 &&
         end_v == max_range) ||
        (stride_v == -1 &&
         begin_v == max_range &&
         end_v == 0)) {
      // Quick path, do not slice this dimension.
      oshape[i] = dshape[i];
      continue;
    }
    // Normal path, require the shape to be concrete integer.
    // Require concrete integer as symbolic inference of min/max
    // can get complicated and not very helpful.
    const int64_t* p_dim_size = as_const_int(dshape[i]);
    CHECK(p_dim_size)
        << "strided_slice requires sliced dimension to be concrete int";
    int64_t dim_size = p_dim_size[0];
    begin_v = (begin_v < 0) ? dim_size + begin_v : begin_v;
    end_v = (end_v < 0) ? dim_size + end_v : end_v;

    int64_t slice_range, step;
    if (stride_v < 0) {
      if (end_v < -1) end_v = -1;
      CHECK_LT(end_v, begin_v)
          << "strided_slice get empty slice at axis " << i;
      begin_v = std::min(dim_size - 1, begin_v);
      slice_range = begin_v - end_v;
      step = -stride_v;
    } else {
      if (begin_v < 0) begin_v = 0;
      CHECK_GE(stride_v, 0);
      CHECK_LT(begin_v, end_v)
          << "strided_slice get empty slice at axis " << i;
      end_v = std::min(dim_size, end_v);
      slice_range = end_v - begin_v;
      step = stride_v;
    }
    oshape[i] = make_const(dshape[i].type(), (slice_range + step - 1) / step);
  }
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}


Array<Array<Layout> > StridedSliceInferCorrectLayout(
    const Attrs& attrs,
    const Array<Layout>& new_in_layouts,
    const Array<Layout>& old_in_layouts,
    const Array<Array<IndexExpr>>& old_in_shapes) {
  CHECK(old_in_layouts.defined());
  CHECK_EQ(old_in_layouts.size(), 1);
  CHECK(old_in_shapes.defined());
  CHECK_EQ(old_in_shapes.size(), 1);

  auto layout = old_in_layouts[0];
  if (layout.defined() && new_in_layouts.defined()) {
    CHECK_EQ(new_in_layouts.size(), 1);
    auto new_layout = new_in_layouts[0];
    auto shape = old_in_shapes[0];

    // NOTE: Discard "const" qualifier here.
    auto *params = const_cast<StridedSliceAttrs*>(attrs.as<StridedSliceAttrs>());

    Array<Integer> new_begin, new_end;

    for (size_t i = 0; i < params->begin.size(); i++) {
      const LayoutAxis& axis = layout[i];
      if (!axis.IsPrimal()) {
        // original layout that contains splitted axes is not supported
        return {{Layout::Undef()}, {Layout::Undef()}};
      }
      auto factor = new_layout.FactorOf(axis);
      if (factor == -1) {
        new_begin.push_back(params->begin[i]);
        new_end.push_back(params->end[i]);
      } else {
        if (params->strides.defined() && i < params->strides.size()) {
          auto stride = params->strides[i];
          // arbitrary stride is not supported
          if (stride.defined() && stride->value != 1) {
            return {{Layout::Undef()}, {Layout::Undef()}};
          }
        }
        int64_t begin = params->begin[i].defined() ? params->begin[i]->value : 0;
        int64_t end = params->end[i].defined() ? params->end[i]->value :
            shape[i].as<IntImm>()->value;
        if (begin % factor || end % factor) {
          // transform to original layout
          return {{Layout::Undef()}, {Layout::Undef()}};
        }
        new_begin.push_back(tvm::Integer(begin / factor));
        new_end.push_back(tvm::Integer(end / factor));
      }
    }
    layout = new_layout;
    params->begin = new_begin;
    params->end = new_end;
  }
  return {{layout}, {layout}};
}


// Positional relay function to create StridedSlice operator used by frontend FFI.
Expr MakeStridedSlice(Expr data,
                      Array<Integer> begin,
                      Array<Integer> end,
                      Array<Integer> strides) {
  auto attrs = make_node<StridedSliceAttrs>();
  attrs->begin = std::move(begin);
  attrs->end = std::move(end);
  attrs->strides = std::move(strides);
  static const Op& op = Op::Get("strided_slice");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

Array<Tensor> StridedSliceCompute(const Attrs& attrs,
                                  const Array<Tensor>& inputs,
                                  const Type& out_type,
                                  const Target& target) {
  const StridedSliceAttrs *param = attrs.as<StridedSliceAttrs>();
  CHECK(param != nullptr);
  return Array<Tensor>{
    topi::strided_slice(inputs[0], param->begin, param->end, param->strides)
  };
}


TVM_REGISTER_API("relay.op._make.strided_slice")
.set_body_typed(MakeStridedSlice);


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
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(4)
.set_attrs_type_key("relay.attrs.StridedSliceAttrs")
.add_type_rel("StridedSlice", StridedSliceRel)
.set_attr<FTVMCompute>("FTVMCompute", StridedSliceCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", StridedSliceInferCorrectLayout);


// relay.split
TVM_REGISTER_NODE_TYPE(SplitAttrs);

bool SplitRel(const Array<Type>& types,
              int num_inputs,
              const Attrs& attrs,
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
  CHECK_LT(axis, data->shape.size())
    << "axis should be within the input dimension range.";
  CHECK_GE(axis, 0)
    << "axis should be within the input dimension range.";

  if (const IntImm* sections = param->indices_or_sections.as<IntImm>()) {
    CHECK(reporter->Assert(data->shape[axis] %
                           sections->value == make_zero(Int(64))))
        << "indices_or_sections need to be able to divide input.shape[axis]";
    std::vector<Type> fields;
    for (int i = 0; i < sections->value; ++i) {
        std::vector<IndexExpr>&& oshape = AsVector(data->shape);
        oshape[axis] /= int32_t(sections->value);
        auto vec_type = TensorTypeNode::make(oshape, data->dtype);
        fields.push_back(vec_type);
    }
    reporter->Assign(types[1], TupleTypeNode::make(Array<Type>(fields)));
  } else {
    auto indices = param->indices_or_sections.as<ArrayNode>()->data;
    auto begin = IndexExpr(make_zero(Int(32)));
    std::vector<Type> fields;
    for (unsigned int i = 0; i < indices.size(); ++i) {
      CHECK(reporter->Assert(IndexExpr(indices[i]) > begin))
          << "indices_or_sections need to be a sorted ascending list";
      std::vector<IndexExpr>&& oshape = AsVector(data->shape);
      oshape[axis] = IndexExpr(indices[i]) - begin;
      begin = IndexExpr(indices[i]);
      auto vec_type = TensorTypeNode::make(oshape, data->dtype);
      fields.push_back(vec_type);
    }
    CHECK(reporter->Assert(begin < data->shape[axis]))
        << "The sum of sections must match the input.shape[axis]";
    std::vector<IndexExpr>&& oshape = AsVector(data->shape);
    oshape[axis] = data->shape[axis] - begin;
    auto vec_type = TensorTypeNode::make(oshape, data->dtype);
    fields.push_back(vec_type);
    reporter->Assign(types[1], TupleTypeNode::make(Array<Type>(fields)));
  }
  return true;
}

Array<Tensor> SplitCompute(const Attrs& attrs,
                           const Array<Tensor>& inputs,
                           const Type& out_type,
                           const Target& target) {
  const auto param = attrs.as<SplitAttrs>();
  CHECK(param != nullptr);

  if (const IntImm* sections = param->indices_or_sections.as<IntImm>()) {
    int64_t num_sections = sections->value;
    return Array<Tensor>{
      topi::split_sections(inputs[0], num_sections, param->axis) };
  } else {
    auto indices = Downcast<Array<Integer> >(param->indices_or_sections);
    return Array<Tensor>{ topi::split(inputs[0], indices, param->axis) };
  }
}

Expr MakeSplit(Expr data,
               NodeRef indices_or_sections,
               int axis) {
  auto attrs = make_node<SplitAttrs>();
  attrs->axis = axis;
  attrs->indices_or_sections = std::move(indices_or_sections);
  static const Op& op = Op::Get("split");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.split")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    if (args.type_codes[1] == kDLInt) {
      *rv = MakeSplit(args[0], make_const(Int(64), int64_t(args[1])), args[2]);
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
.set_attrs_type_key("relay.attrs.SplitAttrs")
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
bool SliceLikeRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
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

  const Array<IndexExpr> dshape = data->shape;
  const Array<IndexExpr> target_shape = target->shape;
  std::vector<IndexExpr>&& oshape = AsVector(dshape);

  if (!param->axes.defined()) {
    for (size_t i = 0; i < dshape.size(); ++i) {
      if (i < target_shape.size()) {
        oshape[i] = target_shape[i];
        CHECK(reporter->Assert(oshape[i] <= dshape[i]))
          << "End index of axis " << i << " exceeds input shape: "
          << oshape[i] << " vs " << dshape[i];
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
        << "Axis " << axis << " exceeds dimension "
        << target_shape.size() << " of target_shape.";
      oshape[axis] = target_shape[axis];
      CHECK(reporter->Assert(oshape[axis] <= dshape[axis]))
        << "End index of axis " << axis << " exceeds input shape: "
        << oshape[axis] << " vs " << dshape[axis];
    }
  }

  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}


Expr MakeSliceLike(Expr data,
                   Expr shape_like,
                   Array<Integer> axes) {
  auto attrs = make_node<SliceLikeAttrs>();
  attrs->axes = std::move(axes);
  static const Op& op = Op::Get("slice_like");
  return CallNode::make(op, {data, shape_like}, Attrs(attrs), {});
}

Array<Tensor> SliceLikeCompute(const Attrs& attrs,
                               const Array<Tensor>& inputs,
                               const Type& out_type,
                               const Target& target) {
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
        CHECK_LE(topi::GetConstInt(end_idx[i]),
                 topi::GetConstInt(src_shape[i]))
          << "End index of axis " << i << " exceeds input shape: "
          << topi::GetConstInt(end_idx[i]) << " vs "
          << topi::GetConstInt(src_shape[i]);
      }
    }
  } else {
    for (int axis : param->axes) {
      if (axis < 0) {
        axis = static_cast<int>(src_shape.size()) + axis;
      }
      end_idx.Set(axis, target_shape[axis]);
      CHECK_LE(topi::GetConstInt(end_idx[axis]),
               topi::GetConstInt(src_shape[axis]))
        << "End index of axis " << axis << " exceeds input shape: "
        << topi::GetConstInt(end_idx[axis]) << " vs "
        << topi::GetConstInt(src_shape[axis]);
    }
  }
  return Array<Tensor>{
    topi::strided_slice(inputs[0],
                        GetIntArray(begin_idx),
                        GetIntArray(end_idx),
                        GetIntArray(strides))
  };
}


TVM_REGISTER_API("relay.op._make.slice_like")
.set_body_typed(MakeSliceLike);


RELAY_REGISTER_OP("slice_like")
.describe(R"code(Slice the first input respect to the second input.
)code" TVM_ADD_FILELINE)
  .set_attrs_type_key("relay.attrs.SlicelikeAttrs")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("shape_like", "Tensor", "Shape tensor.")
.set_support_level(10)
.add_type_rel("SliceLike", SliceLikeRel)
.set_attr<FTVMCompute>("FTVMCompute", SliceLikeCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);

// relay.layout_transform
Array<Tensor> LayoutTransformCompute(const Attrs& attrs,
                                     const Array<Tensor>& inputs,
                                     const Type& out_type,
                                     const Target& target) {
  const auto* param = attrs.as<LayoutTransformAttrs>();
  CHECK(param != nullptr);
  return Array<Tensor>{
    topi::layout_transform(inputs[0], param->src_layout, param->dst_layout)
  };
}

bool LayoutTransformRel(const Array<Type>& types,
                        int num_inputs,
                        const Attrs& attrs,
                        const TypeReporter& reporter) {
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  const LayoutTransformAttrs* params = attrs.as<LayoutTransformAttrs>();

  Layout src_layout(params->src_layout);
  Layout dst_layout(params->dst_layout);

  CHECK(src_layout.defined() && dst_layout.defined())
    << "cannot convert from/to undefined layout";

  auto layout_converter = BijectiveLayoutNode::make(src_layout, dst_layout);
  CHECK(layout_converter.defined())
    << "cannot convert from " << params->src_layout << " to " << params->dst_layout;

  const auto& out_shape = layout_converter.ForwardShape(data->shape);
  reporter->Assign(types[1], TensorTypeNode::make(out_shape, data->dtype));
  return true;
}

Expr MakeLayoutTransform(Expr data,
                         std::string src_layout,
                         std::string dst_layout) {
  auto attrs = make_node<LayoutTransformAttrs>();
  attrs->src_layout = std::move(src_layout);
  attrs->dst_layout = std::move(dst_layout);
  static const Op& op = Op::Get("layout_transform");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.layout_transform")
.set_body_typed(MakeLayoutTransform);

RELAY_REGISTER_OP("layout_transform")
.describe(R"code(Transform the input data layout.

For transforming from NCHW to N16cHWC, the `__layout_transform__` operator reshapes
the input array by output[n, c, h, w, C] = data[n, C*16+c, h, w]

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.LayoutTransformAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.add_type_rel("layout_transform", LayoutTransformRel)
.set_support_level(5)
.set_attr<FTVMCompute>("FTVMCompute", LayoutTransformCompute);


/* relay._contrib_reverse_reshape */
Expr MakeReverseReshape(Expr data,
                        Array<Integer> newshape) {
  auto attrs = make_node<ReshapeAttrs>();
  attrs->newshape = std::move(newshape);
  attrs->reverse = true;
  static const Op& op = Op::Get("_contrib_reverse_reshape");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make._contrib_reverse_reshape")
.set_body_typed(MakeReverseReshape);

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
.set_attrs_type_key("relay.attrs.ReshapeAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(10)
.add_type_rel("Reshape", ReshapeRel)
.set_attr<FTVMCompute>("FTVMCompute", ReshapeCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);

// gather_nd operator
bool GatherNDRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  // `types` contains: [data, indices, result]
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* indices = types[1].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "GatherND: expect input data type to be TensorType but get "
        << types[0];
    return false;
  }
  if (indices == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
        << "GatherND: expect indices type to be TensorType but get "
        << types[1];
    return false;
  }
  const size_t ndim = data->shape.size();
  const IntImm* mdim = data->shape[0].as<IntImm>();
  const size_t kdim = indices->shape.size() - 1;
  CHECK(size_t(mdim->value) <= ndim)
        << "GatherND: indices shape does satisfy.";

  Array<IndexExpr> oshape;
  for (size_t i = 1; i < kdim + 1; ++i)
      oshape.push_back(indices->shape[i]);
  for (size_t i = mdim->value; i < ndim; ++i)
      oshape.push_back(data->shape[i]);
  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Array<Tensor> GatherNDCompute(const Attrs& attrs,
                              const Array<Tensor>& inputs,
                              const Type& out_type,
                              const Target& target) {
  return { topi::gather_nd(inputs[0], inputs[1]) };
}

Expr MakeGatherND(Expr data,
                  Expr indices) {
  static const Op& op = Op::Get("gather_nd");
  return CallNode::make(op, {data, indices}, {});
}

TVM_REGISTER_API("relay.op._make.gather_nd")
.set_body_typed(MakeGatherND);

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

bool SequenceMaskRel(const Array<Type>& types,
                     int num_inputs,
                     const Attrs& attrs,
                     const TypeReporter& reporter) {
  // `types` contains: [data, valid_length, result]
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  const auto* valid_length = types[1].as<TensorTypeNode>();
  CHECK(valid_length != nullptr);
  reporter->Assign(types[2], types[0]);
  return true;
}

Array<Tensor> SequenceMaskCompute(const Attrs& attrs,
                                  const Array<Tensor>& inputs,
                                  const Type& out_type,
                                  const Target& target) {
  const auto* param = attrs.as<SequenceMaskAttrs>();
  CHECK(param != nullptr);
  return Array<Tensor>{ topi::sequence_mask(inputs[0], inputs[1], param->mask_value, param->axis) };
}

Expr MakeSequenceMask(Expr data,
                      Expr valid_length,
                      double mask_value,
                      int axis) {
  auto attrs = make_node<SequenceMaskAttrs>();
  attrs->mask_value = std::move(mask_value);
  attrs->axis = std::move(axis);
  static const Op& op = Op::Get("sequence_mask");
  return CallNode::make(op, {data, valid_length}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.sequence_mask")
.set_body_typed(MakeSequenceMask);

RELAY_REGISTER_OP("sequence_mask")
.describe(R"code(Sets all elements outside the expected length of the sequence to a constant value.

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
.set_attrs_type_key("relay.attrs.SequenceMaskAttrs")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("valid_length", "Tensor", "The real (valid) length of each sequence.")
.set_support_level(10)
.add_type_rel("SequenceMask", SequenceMaskRel)
.set_attr<FTVMCompute>("FTVMCompute", SequenceMaskCompute)
.set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace relay
}  // namespace tvm
