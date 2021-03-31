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

#include <tvm/ir/error.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/nn.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/transform.h>

#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../../transforms/pass_utils.h"
#include "../../transforms/pattern_utils.h"
#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {
using tir::IntImmNode;

// relay.cast
TVM_REGISTER_NODE_TYPE(CastAttrs);

bool CastRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
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
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "cast: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* dtype_like = types[1].as<TensorTypeNode>();
  if (dtype_like == nullptr) {
    ICHECK(types[1].as<IncompleteTypeNode>())
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
  ICHECK(param != nullptr);
  DataType dtype = param->dtype;
  return {topi::reinterpret(inputs[0], dtype)};
}

Expr MakeReinterpret(Expr data, DataType dtype) {
  auto attrs = make_object<CastAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("reinterpret");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay._make.reinterpret").set_body_typed(MakeReinterpret);

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
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "expand_dims: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<ExpandDimsAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  const int axis = param->axis;
  const int num_newaxis = param->num_newaxis;
  ICHECK(num_newaxis >= 0) << "expand_dims only accepts `num_newaxis >= 0`"
                           << ", but got num_newaxis = " << num_newaxis;
  ICHECK(-ndim - 1 <= axis && axis <= ndim)
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
  ICHECK(param != nullptr);
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
    .describe(R"code(Insert `num_newaxis` axes at the position given by `axis`

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
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 2);
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "cast: expect input type to be TupleType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<StackAttrs>();
  const auto& first = Downcast<TensorType>(tensor_tuple->fields[0]);
  const int ndim = static_cast<int>(first->shape.size());

  // Sanity check: axis
  int axis = param->axis;
  ICHECK(-(ndim + 1) <= axis && axis < ndim + 1)
      << "stack only accepts `axis` in [-(ndim+1), ndim+1)"
      << ", but got axis = " << axis << ", and ndim = " << ndim;
  axis = axis < 0 ? ndim + axis + 1 : axis;

  // Sanity check: ndim and dtype.
  const DataType dtype = first->dtype;
  for (const Type& ele : tensor_tuple->fields) {
    const auto& e = Downcast<TensorType>(ele);
    int e_ndim = static_cast<int>(e->shape.size());
    const DataType& e_dtype = e->dtype;
    ICHECK_EQ(e_ndim, ndim) << "relay.stack requires all tensors have the same ndim";
    ICHECK_EQ(e_dtype, dtype) << "relay.stack requires all tensors have the same dtype";
    for (size_t j = 0; j < first->shape.size(); ++j) {
      if (j == static_cast<size_t>(axis)) continue;
      if (first->shape[j].as<AnyNode>() || e->shape[j].as<AnyNode>() ||
          reporter->AssertEQ(first->shape[j], e->shape[j]))
        continue;
      throw CompileError(
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
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "transpose: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<TransposeAttrs>();
  const int ndim = data->shape.size();
  const Array<Integer>& axes = param->axes;
  // check dimension match
  ICHECK(!axes.defined() || static_cast<int>(axes.size()) == ndim)
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
      ICHECK(-ndim <= axis && axis < ndim)
          << "transpose only allows each `axis` in `axes` in range [-data.ndim, data.ndim)"
          << ", but got axis = " << axis << ", and data.ndim = " << ndim;
      axis = axis < 0 ? axis + ndim : axis;
      // sanity check for duplication
      ICHECK(!axis_used[axis]) << "Duplicate axes in transpose: " << axis;
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

Array<Array<Layout>> TransposeInferCorrectLayout(const Attrs& attrs,
                                                 const Array<Layout>& new_in_layouts,
                                                 const Array<Layout>& old_in_layouts,
                                                 const Array<tvm::relay::Type>& old_in_types) {
  // Discard "const" qualifier.
  auto* params = const_cast<TransposeAttrs*>(attrs.as<TransposeAttrs>());
  ICHECK(params != nullptr);

  std::string in_layout_str = "";
  std::string out_layout_str = "";

  // Infer the input layout string and update the axes.
  if (old_in_layouts.defined() && old_in_layouts[0].defined()) {
    ICHECK_EQ(old_in_layouts.size(), 1);
    auto old_layout = old_in_layouts[0];
    Array<Integer> old_axes = params->axes;

    // Deal with default axes and negative axes.
    if (!old_axes.defined() || old_axes.size() == 0) {
      for (int i = old_layout.ndim() - 1; i >= 0; --i) {
        old_axes.push_back(i);
      }
    }
    for (size_t i = 0; i < old_axes.size(); ++i) {
      int axis = static_cast<int>(old_axes[i]->value);
      if (axis < 0) {
        int pos_axis = static_cast<int>(old_layout.ndim()) + axis;
        old_axes.Set(i, pos_axis);
      }
    }

    if (new_in_layouts.defined() && new_in_layouts[0].defined()) {
      ICHECK_EQ(new_in_layouts.size(), 1);
      auto new_layout = new_in_layouts[0];

      // Update the axes based on the new layout.
      Array<Integer> new_axes = Array<Integer>();
      for (auto axis : old_axes) {
        auto new_axis = new_layout.IndexOf(old_layout[axis->value]);
        if (new_axis == -1) {  // Cannot find the target axis in the new layout.
          new_axes.clear();
          break;
        }
        new_axes.push_back(new_axis);
      }
      if (new_axes.defined() && new_axes.size() == new_layout.ndim()) {
        params->axes = std::move(new_axes);
        in_layout_str = new_layout.name();
      }
    }

    // If the input layout string cannot be determined, propagate the old layout.
    if (in_layout_str == "") {
      params->axes = std::move(old_axes);
      in_layout_str = old_layout.name();
    }
  }

  // Infer the output layout string based on the input layout and the axes.
  if (in_layout_str != "") {
    for (auto axis : params->axes) {
      ICHECK_LT(axis->value, in_layout_str.length());
      out_layout_str += in_layout_str[axis->value];
    }
    try {
      return Array<Array<Layout>>({{Layout(in_layout_str)}, {Layout(out_layout_str)}});
    } catch (const tvm::Error& e) {
      // If the layout string is invalid for any reason, give up.
      return Array<Array<Layout>>({{Layout::Undef()}, {Layout::Undef()}});
    }
  }
  return Array<Array<Layout>>({{Layout::Undef()}, {Layout::Undef()}});
}

Array<te::Tensor> TransposeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                   const Type& out_type) {
  const auto* param = attrs.as<TransposeAttrs>();
  ICHECK(param != nullptr);
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
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", TransposeInferCorrectLayout)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

/* relay.reshape */
TVM_REGISTER_NODE_TYPE(ReshapeAttrs);
TVM_REGISTER_NODE_TYPE(ReshapeLikeAttrs);

Array<IndexExpr> InferNewShape(const Array<IndexExpr>& data_shape, const Attrs& attrs,
                               bool reverse) {
  const auto* param = attrs.as<ReshapeAttrs>();
  Array<IndexExpr> oshape;
  Array<IndexExpr> ishape;
  Array<Integer> newshape;

  if (reverse) {
    ishape.Assign(data_shape.rbegin(), data_shape.rend());
    newshape.Assign(param->newshape.rbegin(), param->newshape.rend());
  } else {
    ishape = data_shape;
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
      ICHECK_LT(src_idx, ishape.size());
      used_input_dims.insert(src_idx);
      used_output_dims.insert(oshape.size());
      oshape.push_back(ishape[src_idx++]);
    } else if (svalue == -1) {
      // inference based on rest
      ICHECK_LT(infer_idx, 0) << "One and only one dim can be inferred";
      infer_idx = i;
      oshape.push_back(1);
      ++src_idx;
    } else if (svalue == -2) {
      // copy all remaining dims from source
      while (src_idx < ishape.size()) {
        used_input_dims.insert(src_idx);
        used_output_dims.insert(oshape.size());
        oshape.push_back(ishape[src_idx++]);
      }
    } else if (svalue == -3) {
      // merge two dims from source
      ICHECK_LT(src_idx + 1, ishape.size());
      used_input_dims.insert(src_idx);
      IndexExpr d1 = ishape[src_idx++];
      used_input_dims.insert(src_idx);
      IndexExpr d2 = ishape[src_idx++];
      used_output_dims.insert(oshape.size());
      if (d1.as<AnyNode>() || d2.as<AnyNode>()) {
        oshape.push_back(Any());
      } else {
        oshape.push_back(d1 * d2);
      }
    } else if (svalue == -4) {
      // split the source dim s into two dims
      // read the left dim and then the right dim (either can be -1)
      ICHECK_LT(i + 2, newshape.size());
      ICHECK_LT(src_idx, ishape.size());
      used_input_dims.insert(src_idx);
      IndexExpr d0 = ishape[src_idx++];
      Integer d1 = newshape[++i];
      Integer d2 = newshape[++i];
      if (d1->value == -1) {
        ICHECK_NE(d2->value, -1) << "Split dims cannot both be -1.";
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
      LOG(FATAL) << "Unsupported special value: " << svalue;
    }
  }

  if (infer_idx >= 0) {
    IndexExpr infer_dim = 1;
    for (size_t i = 0; i < ishape.size(); ++i) {
      if (used_input_dims.count(i) != 0) {
        continue;
      }
      if (ishape[i].as<AnyNode>()) {
        infer_dim = Any();
        break;
      }
      infer_dim *= ishape[i];
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
    arith::Analyzer ana;
    infer_dim = ana.Simplify(infer_dim);
    oshape.Set(infer_idx, infer_dim);
  }

  return oshape;
}

bool ReshapeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  // types: [data, result]
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "reshape: expect input type to be TensorType but get " << types[0];
    return false;
  }

  const auto& oshape = InferNewShape(data->shape, attrs, false);

  // Verify that the sum of dimensions in the output shape is the sum of
  // dimensions in the input shape
  Array<IndexExpr> data_shape;
  data_shape = data->shape;

  bool found_dynamic = false;
  int64_t oshape_sum = 1;
  for (auto& x : oshape) {
    // Check if we have a dynamic shape. If we do, we can't verify if the
    // reshape is valid. Dynamic shapes are marker by using Any, but can also
    // occur from SizeVar's. In the case of SizeVar, the shape expression can
    // be an AST. We can't easily check if we have an AST because of a ShapeVar
    // or some other reason, so our check for dynamic shape is just if we can
    // convert the shape to in integer or not.
    if (!x->IsInstance<tvm::Integer::ContainerType>()) {
      found_dynamic = true;
      break;
    }
    oshape_sum *= Downcast<tvm::Integer>(x)->value;
  }
  int64_t data_shape_sum = 1;
  for (auto& x : data_shape) {
    if (!x->IsInstance<tvm::Integer::ContainerType>()) {
      found_dynamic = true;
      break;
    }
    data_shape_sum *= Downcast<tvm::Integer>(x)->value;
  }
  if (!found_dynamic) {
    ICHECK_EQ(oshape_sum, data_shape_sum)
        << "Input tensor shape and reshaped shape are not compatible";
  }

  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

bool ReverseReshapeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  // types: [data, result]
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "reshape: expect input type to be TensorType but get " << types[0];
    return false;
  }

  const auto& oshape = InferNewShape(data->shape, attrs, true);

  // Verify that the sum of dimensions in the output shape is the sum of
  // dimensions in the input shape
  Array<IndexExpr> data_shape;
  data_shape.Assign(data->shape.rbegin(), data->shape.rend());

  bool found_dynamic = false;
  int64_t oshape_sum = 1;
  for (auto& x : oshape) {
    // Check if we have a dynamic shape. If we do, we can't verify if the
    // reshape is valid. Dynamic shapes are marker by using Any, but can also
    // occur from SizeVar's. In the case of SizeVar, the shape expression can
    // be an AST. We can't easily check if we have an AST because of a ShapeVar
    // or some other reason, so our check for dynamic shape is just if we can
    // convert the shape to in integer or not.
    if (!x->IsInstance<tvm::Integer::ContainerType>()) {
      found_dynamic = true;
      break;
    }
    oshape_sum *= Downcast<tvm::Integer>(x)->value;
  }
  int64_t data_shape_sum = 1;
  for (auto& x : data_shape) {
    if (!x->IsInstance<tvm::Integer::ContainerType>()) {
      found_dynamic = true;
      break;
    }
    data_shape_sum *= Downcast<tvm::Integer>(x)->value;
  }
  if (!found_dynamic) {
    ICHECK_EQ(oshape_sum, data_shape_sum)
        << "Input tensor shape and reshaped shape are not compatible";
  }

  reporter->Assign(types[1],
                   TensorType(Array<IndexExpr>(oshape.rbegin(), oshape.rend()), data->dtype));
  return true;
}

Array<PrimExpr> infer_reshape_like(const Array<PrimExpr>& lhs_shape,
                                   const Array<PrimExpr>& rhs_shape, const Attrs& attrs) {
  const auto* like_attrs = attrs.as<ReshapeLikeAttrs>();
  CHECK(!like_attrs->lhs_end.defined() || like_attrs->lhs_end.as<IntImmNode>())
      << "lhs_end must be a concrete integer or None";
  CHECK(!like_attrs->rhs_end.defined() || like_attrs->rhs_end.as<IntImmNode>())
      << "rhs_end must be a concrete integer or None";

  int64_t lhs_shape_size = static_cast<int64_t>(lhs_shape.size());
  int64_t rhs_shape_size = static_cast<int64_t>(rhs_shape.size());
  int64_t lhs_begin = static_cast<int64_t>(like_attrs->lhs_begin);
  int64_t lhs_end =
      like_attrs->lhs_end.defined() ? like_attrs->lhs_end.as<IntImmNode>()->value : lhs_shape_size;
  int64_t rhs_begin = static_cast<int64_t>(like_attrs->rhs_begin);
  int64_t rhs_end =
      like_attrs->rhs_end.defined() ? like_attrs->rhs_end.as<IntImmNode>()->value : rhs_shape_size;

  // handle negative axes
  lhs_begin = lhs_begin < 0 ? lhs_begin + lhs_shape_size : lhs_begin;
  lhs_end = lhs_end < 0 ? lhs_end + lhs_shape_size : lhs_end;
  rhs_begin = rhs_begin < 0 ? rhs_begin + rhs_shape_size : rhs_begin;
  rhs_end = rhs_end < 0 ? rhs_end + rhs_shape_size : rhs_end;

  Array<PrimExpr> shape_like;
  for (auto i = 0; i < lhs_begin; i++) {
    shape_like.push_back(lhs_shape[i]);
  }
  for (auto i = rhs_begin; i < rhs_end; i++) {
    shape_like.push_back(rhs_shape[i]);
  }
  for (auto i = lhs_end; i < lhs_shape_size; i++) {
    shape_like.push_back(lhs_shape[i]);
  }
  return shape_like;
}

Array<te::Tensor> ReshapeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {
  // Quick path for reshape_like
  if (!attrs.as<ReshapeAttrs>()) {
    ICHECK(attrs.as<ReshapeLikeAttrs>() != nullptr);
    auto shape_like = infer_reshape_like(inputs[0]->shape, inputs[1]->shape, attrs);
    return {topi::reshape(inputs[0], shape_like)};
  }

  const auto* out_ttype = out_type.as<TensorTypeNode>();
  ICHECK(out_ttype != nullptr);
  Array<IndexExpr> newshape;
  bool newshape_has_any = false;
  for (auto val : out_ttype->shape) {
    if (val->IsInstance<tir::AnyNode>() || val->IsInstance<tir::VarNode>()) {
      newshape_has_any = true;
      break;
    } else {
      newshape.push_back(val);
    }
  }

  if (newshape_has_any) {
    newshape = InferNewShape(inputs[0]->shape, attrs, false);
  }
  return {topi::reshape(inputs[0], newshape)};
}

Expr MakeReshape(Expr data, Array<Integer> newshape) {
  auto attrs = make_object<ReshapeAttrs>();
  attrs->newshape = std::move(newshape);
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
  ICHECK(attrs.as<ReshapeLikeAttrs>() != nullptr);
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* reshape_like = types[1].as<TensorTypeNode>();
  if (reshape_like == nullptr) {
    return false;
  }
  auto shape_like = infer_reshape_like(data->shape, reshape_like->shape, attrs);
  // Only check When input data has static shape.
  bool is_static_shape = true;
  for (size_t i = 0; i < data->shape.size(); ++i) {
    if (!data->shape[i].as<IntImmNode>()) {
      is_static_shape = false;
      break;
    }
  }
  auto output_type = TensorType(shape_like, data->dtype);
  if (is_static_shape) {
    ICHECK(reporter->AssertEQ(data->Size(), output_type->Size()))
        << "Reshape inputs size should be compatible.";
  }
  reporter->Assign(types[2], output_type);
  return true;
}

Expr MakeReshapeLike(Expr lhs, Expr rhs, int lhs_begin, Integer lhs_end, int rhs_begin,
                     Integer rhs_end) {
  auto attrs = make_object<ReshapeLikeAttrs>();
  attrs->lhs_begin = std::move(lhs_begin);
  attrs->lhs_end = std::move(lhs_end);
  attrs->rhs_begin = std::move(rhs_begin);
  attrs->rhs_end = std::move(rhs_end);
  static const Op& op = Op::Get("reshape_like");
  return Call(op, {lhs, rhs}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.reshape_like").set_body_typed(MakeReshapeLike);

RELAY_REGISTER_OP("reshape_like")
    .describe(R"code(Reshapes the input array by the size of another array.
For an input array with shape ``(d1, d2, ..., dk)``, `reshape_like` operation reshapes
the input array into an output array with the same shape as the second input array.
.. note::
    Sizes for both array should be compatible.
Example::

  data.shape == (1, 2, 3, 4)
  shape_like.shape == (6, 2, 2, 3)

  ret = reshape_like(data, shape_like, lhs_begin=1, rhs_end=3)
  ret.shape == (1, 6, 2, 2)
)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReshapeLikeAttrs>()
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
  ICHECK_EQ(num_inputs, 1);
  auto tt = types[0].as<TensorTypeNode>();

  if (tt == nullptr) {
    return false;
  }

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
  ICHECK_EQ(num_inputs, 3);
  ICHECK_EQ(types.size(), 4);
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
  ICHECK(indices->dtype.is_int()) << "indices of take must be tensor of integer";
  const auto param = attrs.as<ScatterAttrs>();
  ICHECK(param != nullptr);
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

// Scatter_add
TVM_REGISTER_NODE_TYPE(ScatterAddAttrs);

// Scatter Add
bool ScatterAddRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(num_inputs, 3);
  ICHECK_EQ(types.size(), 4);
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
  ICHECK(indices->dtype.is_int()) << "indices of scatter_add must be tensor of integer";
  const auto param = attrs.as<ScatterAddAttrs>();
  ICHECK(param != nullptr);
  reporter->Assign(types[3], TensorType(data->shape, data->dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op._make.scatter_add")
    .set_body_typed([](Expr data, Expr indices, Expr updates, int axis) {
      auto attrs = make_object<ScatterAddAttrs>();
      attrs->axis = std::move(axis);
      static const Op& op = Op::Get("scatter_add");
      return Call(op, {data, indices, updates}, Attrs(attrs), {});
    });

RELAY_REGISTER_OP("scatter_add")
    .describe(
        R"doc(Update data by adding values in updates at positions defined by indices)doc" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input data tensor.")
    .add_argument("indicies", "Tensor", "The indicies location tensor.")
    .add_argument("updates", "Tensor", "The values to update the input with.")
    .add_type_rel("ScatterAdd", ScatterAddRel)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_support_level(10);

// scatter_nd operator
TVM_REGISTER_NODE_TYPE(ScatterNDAttrs);

bool ScatterNDRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  // `types` contains: [data, indices, result]
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* indices = types[1].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "ScatterND: expect input data type to be TensorType but got " << types[0];
    return false;
  }
  if (indices == nullptr) {
    ICHECK(types[1].as<IncompleteTypeNode>())
        << "ScatterND: expect indices type to be TensorType but got " << types[1];
    return false;
  }
  ICHECK(indices->dtype.is_int()) << "ScatterND: indices must be a tensor of integers.";
  const auto out_shape = attrs.as<ScatterNDAttrs>()->out_shape;
  const IntImmNode* mdim = indices->shape[0].as<IntImmNode>();
  const size_t kdim = indices->shape.size() - 1;
  const size_t ndim = out_shape.size();
  ICHECK_LE(size_t(mdim->value), ndim)
      << "ScatterND: Given data with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}), and indices "
         "with shape (M, Y_0, ..., Y_{K-1}), M must be less than or equal to N.";
  // Indices: (M, Y_0, .. Y_{K-1}) data: (Y_0, .. Y_{K-1}, ...), verify Y's.
  for (size_t i = 0; i < kdim; i++) {
    reporter->AssertEQ(indices->shape[i + 1], data->shape[i]);
  }

  std::vector<IndexExpr> oshape;
  for (auto& x : out_shape) {
    oshape.push_back(x);
  }

  // data: (Y_0, .. Y_{K-1}, X_M, .. X_{N-1}) out: (X_0, .. X_{N-1}), verify X_M to X_{N-1}
  for (size_t i = mdim->value; i < ndim; i++) {
    reporter->AssertEQ(data->shape[i - mdim->value + kdim], oshape[i]);
  }

  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeScatterND(Expr data, Expr indices, const Array<Integer> out_shape) {
  auto attrs = make_object<ScatterNDAttrs>();
  attrs->out_shape = out_shape;
  static const Op& op = Op::Get("scatter_nd");
  return Call(op, {data, indices}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.scatter_nd").set_body_typed(MakeScatterND);

// scatter_nd operator has extern schedules for CPU and GPU devices.
// Fusing extern schedules with Injective schedules leads to errors.
// So, converting the scatter_nd to Opaque to prevent compilation failures
RELAY_REGISTER_OP("scatter_nd")
    .describe(R"code(Scatter elements or slices from data and store to a tensor
whose shape is defined by indices.

Given data with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}) and indices with shape
(M, Y_0, ..., Y_{K-1}), the output will have shape (X_0, X_1, ..., X_{N-1}).
)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices tensor.")
    .set_support_level(3)
    .add_type_rel("ScatterND", ScatterNDRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

// Take
TVM_REGISTER_NODE_TYPE(TakeAttrs);

bool TakeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
             const TypeReporter& reporter) {
  // `types` contains: [data, indices, result]
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* indices = types[1].as<TensorTypeNode>();
  if (indices == nullptr) {
    return false;
  }
  ICHECK(indices->dtype.is_int()) << "indices of take must be tensor of integer";
  const auto param = attrs.as<TakeAttrs>();
  ICHECK(param != nullptr);

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
  ICHECK_LE(axis, ndim_data) << "axis should be with in data shape"
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
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 2);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  const auto* fill_value = types[0].as<TensorTypeNode>();
  if (fill_value == nullptr) {
    return false;
  }

  DataType out_dtype = param->dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = fill_value->dtype;
  }

  ICHECK_EQ(fill_value->shape.size(), 0)
      << "Fill value should be a scalar but has dimension " << fill_value->shape.size() << ".";

  std::vector<IndexExpr> oshape;
  const Array<Integer>& cshape_array = param->shape.value();
  for (size_t i = 0; i < cshape_array.size(); ++i) {
    oshape.push_back(cshape_array[i]);
  }
  reporter->Assign(types[1], TensorType(oshape, out_dtype));
  return true;
}

Expr MakeFull(Expr fill_value, Array<Integer> shape, DataType dtype) {
  auto attrs = make_object<InitOpAttrs>();
  attrs->dtype = std::move(dtype);
  attrs->shape = std::move(shape);
  static const Op& op = Op::Get("full");
  return Call(op, {fill_value}, Attrs(attrs), {});
}

Array<te::Tensor> FullCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  return {topi::full(out_ttype->shape, out_ttype->dtype, inputs[0]())};
}

TVM_REGISTER_GLOBAL("relay.op._make.full").set_body_typed(MakeFull);

RELAY_REGISTER_OP("full")
    .describe(R"code(Fill array with scalar value.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<InitOpAttrs>()
    .set_num_inputs(1)
    .add_argument("fill_value", "double", "The value to fill.")
    .set_support_level(3)
    .add_type_rel("Full", FullRel)
    .set_attr<FTVMCompute>("FTVMCompute", FullCompute)
    .set_attr<TOpPattern>("TOpPattern", kElemWise);

bool InitOpRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types = [ret_type]
  ICHECK_EQ(types.size(), 1);

  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  ICHECK(param);

  DataType out_dtype = param->dtype;
  std::vector<IndexExpr> oshape;

  const Array<Integer>& cshape_array = param->shape.value();
  for (size_t i = 0; i < cshape_array.size(); ++i) {
    oshape.push_back(cshape_array[i]);
  }
  reporter->Assign(types[0], TensorType(oshape, out_dtype));
  return true;
}

Expr MakeZeros(Array<Integer> shape, DataType dtype) {
  auto attrs = make_object<InitOpAttrs>();
  attrs->shape = std::move(shape);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("zeros");
  return Call(op, {}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.zeros").set_body_typed(MakeZeros);

RELAY_REGISTER_OP("zeros")
    .describe(R"code(Fill array with zeros.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<InitOpAttrs>()
    .set_num_inputs(0)
    .set_support_level(3)
    .add_type_rel("InitOp", InitOpRel);

Expr MakeOnes(Array<Integer> shape, DataType dtype) {
  auto attrs = make_object<InitOpAttrs>();
  attrs->shape = std::move(shape);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("ones");
  return Call(op, {}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.ones").set_body_typed(MakeOnes);

RELAY_REGISTER_OP("ones")
    .describe(R"code(Fill array with ones.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<InitOpAttrs>()
    .set_num_inputs(0)
    .set_support_level(3)
    .add_type_rel("InitOp", InitOpRel);

bool FullLikeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* fill_value = types[1].as<TensorTypeNode>();
  if (fill_value == nullptr) {
    return false;
  }

  ICHECK_EQ(fill_value->shape.size(), 0)
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
  ICHECK_EQ(types.size(), 4);
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
    ICHECK_GT(num_elem, 0) << "Invalid arange attributes (start, stop, step): " << attrs->start
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
  ICHECK(param != nullptr);
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
    .add_argument("start", "Expr", "Start of interval. The interval includes this value.")
    .add_argument("end", "Expr", "Stop of interval. The interval does not include this value.")
    .add_argument("step", "Expr", "Spacing between values.")
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
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "repeat: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<RepeatAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  const int repeats = param->repeats;
  const int axis = param->axis;
  ICHECK(repeats >= 1) << "repeat only accepts `repeats >= 1`"
                       << ", but got repeats = " << repeats;
  ICHECK(-ndim - 1 <= axis && axis <= ndim)
      << "repeat only accepts `axis` in [-data.ndim - 1, data.ndim]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  const int pivot = axis < 0 ? ndim + axis : axis;
  std::vector<IndexExpr> oshape;
  oshape.reserve(ndim + repeats);
  for (int i = 0; i < pivot; ++i) {
    oshape.emplace_back(data->shape[i]);
  }
  if (data->shape[pivot].as<AnyNode>()) {
    oshape.emplace_back(Any());
  } else {
    oshape.emplace_back(data->shape[pivot] * repeats);
  }
  for (int i = pivot + 1; i < ndim; ++i) {
    oshape.emplace_back(data->shape[i]);
  }
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> RepeatCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  const RepeatAttrs* param = attrs.as<RepeatAttrs>();
  ICHECK(param != nullptr);
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

bool SparseFillEmptyRowsRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                            const TypeReporter& reporter) {
  // types: [sparse_indices, sparse_values, dense_shape, default_value, result]
  ICHECK_EQ(types.size(), 5) << "SparseFillEmptyRowsRel expects 5 inputs but " << types.size()
                             << "provided";
  std::vector<Type> fields;
  auto sparse_indices = types[0].as<TensorTypeNode>();
  auto ndims = sparse_indices->shape[1];
  fields.push_back(TensorType(Array<PrimExpr>{Any(), ndims}, tvm::DataType::Int(64)));
  fields.push_back(TensorType(Array<PrimExpr>{Any()}, tvm::DataType::Int(64)));
  fields.push_back(TensorType(Array<PrimExpr>{Any()}, tvm::DataType::Int(64)));
  reporter->Assign(types[types.size() - 1], TupleType(Array<Type>(fields)));
  return true;
}

Expr MakeSparseFillEmptyRows(Expr sparse_indices, Expr sparse_values, Expr dense_shape,
                             Expr default_value) {
  static const Op& op = Op::Get("sparse_fill_empty_rows");
  return Call(op, {sparse_indices, sparse_values, dense_shape, default_value}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.sparse_fill_empty_rows")
    .set_body_typed(MakeSparseFillEmptyRows);

RELAY_REGISTER_OP("sparse_fill_empty_rows")
    .describe(
        R"code(Fill empty rows of a sparse tensor with a default value.)code" TVM_ADD_FILELINE)
    .set_num_inputs(4)
    .add_argument("sparse_indices", "Tensor",
                  "A 2-D int64 tensor of shape [N, ndims], which specifies the indices of the"
                  "elements in the sparse tensor that contain nonzero values. COO Format")
    .add_argument(
        "sparse_values", "Tensor",
        "A 1-D tensor[N] which supplies the values for each element in indices. COO Format")
    .add_argument("dense_shape", "Tensor",
                  "A 1-D int64 tensor of shape [ndims], which specifies the dense_shape of the"
                  "sparse tensor. Takes a list indicating the number of elements in each "
                  "dimension")
    .add_argument("default_value", "Tensor",
                  "The value to fill for empty rows, with the same type as sparse_values")
    .add_type_rel("sparse_fill_empty_rows", SparseFillEmptyRowsRel)
    .set_support_level(3)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

bool SparseReshapeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  // types: [sparse_indices, prev_shape, new_shape, result]
  ICHECK_EQ(types.size(), 4) << "SparseReshapeRel expects 4 types but " << types.size()
                             << " provided";
  ICHECK_EQ(num_inputs, 3) << "SparseReshapeRel expects 4 inputs but " << num_inputs << " provided";
  auto sparse_indices = types[0].as<TensorTypeNode>();
  auto prev_shape = types[1].as<TensorTypeNode>();
  auto new_shape = types[2].as<TensorTypeNode>();
  if (sparse_indices == nullptr || prev_shape == nullptr || new_shape == nullptr) {
    return false;
  }
  CHECK(sparse_indices->dtype.is_int()) << "sparse_indices must be tensor of integers";
  CHECK(prev_shape->dtype.is_int()) << "prev_shape must be tensor of integers";
  CHECK(new_shape->dtype.is_int()) << "new_shape must be tensor of integers";
  ICHECK_EQ(sparse_indices->shape.size(), 2) << "sparse_indices must be 2-D tensor";
  ICHECK_EQ(prev_shape->shape.size(), 1) << "prev_shape must be 1-D tensor";
  ICHECK_EQ(new_shape->shape.size(), 1) << "new_shape must be 1-D tensor";
  std::vector<Type> fields;
  Array<PrimExpr> new_sparse_indices_shape{sparse_indices->shape[0], new_shape->shape[0]};
  fields.push_back(TensorType(new_sparse_indices_shape, sparse_indices->dtype));
  fields.push_back(TensorType(new_shape->shape, new_shape->dtype));
  reporter->Assign(types[3], TupleType(Array<Type>(fields)));
  return true;
}

Expr MakeSparseReshape(Expr sparse_indices, Expr prev_shape, Expr new_shape) {
  static const Op& op = Op::Get("sparse_reshape");
  return Call(op, {sparse_indices, prev_shape, new_shape}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.sparse_reshape").set_body_typed(MakeSparseReshape);

RELAY_REGISTER_OP("sparse_reshape")
    .describe(R"code(Return new sparse indices of the reshaped tensor
)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("sparse_indices", "Tensor",
                  "A 2-D tensor of shape [N, ndims], which specifies the indices of the"
                  "elements in the sparse tensor that contain nonzero values.  COO Format")
    .add_argument("prev_shape", "Tensor",
                  "A 1-D tensor of shape [ndims], which specifies the previous dense shape of the"
                  "sparse tensor")
    .add_argument("new_shape", "Tensor",
                  "A 1-D tensor of shape [ndims], which specifies the desired dense shape of the"
                  "sparse tensor")
    .add_type_rel("sparse_reshape", SparseReshapeRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_support_level(3);

// meshgrid operator
TVM_REGISTER_NODE_TYPE(MeshgridAttrs);

bool MeshgridRel(const Array<Type>& types, int num_inputs, const Attrs& raw_attrs,
                 const TypeReporter& reporter) {
  // types: [data, result]
  ICHECK_EQ(types.size(), 2);
  const MeshgridAttrs* attrs = raw_attrs.as<MeshgridAttrs>();
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    throw CompileError(ErrorBuilder()
                       << "meshgrid requires a tuple of tensors as the first argument, found "
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
      throw CompileError("relay.meshgrid requires all tensors have the same dtype");
    }
    if (e_ndim == 0) {
      grid_shape.emplace_back(1);
    } else if (e_ndim == 1) {
      grid_shape.emplace_back(e->shape[0]);
    } else {
      throw CompileError("relay.meshgrid requires all tensors be either scalars or 1-D vectors.");
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
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "tile: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<TileAttrs>();
  const size_t ndim = data->shape.size();
  const Array<Integer>& reps = param->reps;
  // check dimension match
  ICHECK(reps.defined()) << "repetition array is not defined. data.ndim = " << ndim;
  const size_t rndim = reps.size();
  for (size_t i = 0; i < rndim; ++i) {
    if (const tvm::tir::IntImmNode* val = reps[i].as<tvm::tir::IntImmNode>()) {
      ICHECK_GT(val->value, 0) << "Tile reps value should always be larger than 0, but get: "
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
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "reverse: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto* param = attrs.as<ReverseAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  const int axis = param->axis;
  ICHECK(-ndim <= axis && axis < ndim)
      << "reverse only accepts `axis` in [-data.ndim, data.ndim - 1]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  reporter->Assign(types[1], types[0]);
  return true;
}

Array<te::Tensor> ReverseCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {
  const ReverseAttrs* param = attrs.as<ReverseAttrs>();
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "reverse_sequence: expect input type to be TensorType but get " << types[0];
    return false;
  }

  const auto* seq_lengths = types[1].as<TensorTypeNode>();
  if (seq_lengths == nullptr) {
    ICHECK(types[1].as<IncompleteTypeNode>())
        << "reverse_sequence: expect input type to be TensorType but get " << types[1];
    return false;
  }

  const int seq_lengths_dim = static_cast<int>(seq_lengths->shape.size());
  ICHECK(seq_lengths_dim == 1) << "For reverse_sequnece, seq_lengths must be a 1D vector";
  ICHECK(seq_lengths->dtype.is_int())
      << "For reverse_sequnece, seq_lengths must be tensor of integer";

  const auto* param = attrs.as<ReverseSequenceAttrs>();
  const int ndim = static_cast<int>(data->shape.size());
  int batch_axis = param->batch_axis;
  ICHECK(-ndim <= batch_axis && batch_axis < ndim)
      << "reverse_sequence only accepts `batch_axis` in [-data.ndim, data.ndim - 1]"
      << ", but got batch_axis = " << batch_axis << ", and data.ndim = " << ndim;

  if (batch_axis < 0) {
    batch_axis = static_cast<int>(data->shape.size()) + batch_axis;
  }
  ICHECK(reporter->Assert(seq_lengths->shape[0] == data->shape[batch_axis]))
      << "For reverse_sequnece seq_lengths size should match with dimension of batch axis"
      << ", but got dimension of batch_axis = " << data->shape[batch_axis]
      << ", and seq_length size = " << seq_lengths->shape[0];

  const int seq_axis = param->seq_axis;
  ICHECK(-ndim <= seq_axis && seq_axis < ndim)
      << "reverse_sequnece only accepts `seq_axis` in [-data.ndim, data.ndim - 1]"
      << ", but got seq_axis = " << seq_axis << ", and data.ndim = " << ndim;

  reporter->Assign(types[2], types[0]);
  return true;
}

Array<te::Tensor> ReverseSequenceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                         const Type& out_type) {
  const ReverseSequenceAttrs* param = attrs.as<ReverseSequenceAttrs>();
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 4U);
  const auto* condition = types[0].as<TensorTypeNode>();
  const auto* x = types[1].as<TensorTypeNode>();
  const auto* y = types[2].as<TensorTypeNode>();

  if (condition == nullptr || x == nullptr || y == nullptr) {
    return false;
  }

  ICHECK_EQ(x->dtype, y->dtype) << "x and y must have the same dtype: " << x->dtype << " vs "
                                << y->dtype;

  auto tensor_ty_condition = GetRef<TensorType>(condition);
  auto tensor_ty_x = GetRef<TensorType>(x);
  auto tensor_ty_y = GetRef<TensorType>(y);

  auto b_ty = ConcreteBroadcast(tensor_ty_x, tensor_ty_y, x->dtype);
  auto ret_ty = ConcreteBroadcast(tensor_ty_condition, b_ty, b_ty->dtype);

  reporter->Assign(types[3], ret_ty);
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

Shapes of condition, x, and y must be broadcastable to a common shape, which
is the output shape of this op. Semantics follow numpy where function.
https://numpy.org/doc/stable/reference/generated/numpy.where.html

Note that all non-zero values are interpreted as True in condition.

Examples::

  x = [[1, 2], [3, 4]]
  y = [[5, 6], [7, 8]]
  cond = [[0, 1], [-1, 0]]
  where(cond, x, y) = [[5, 2], [3, 8]]


  cond = [[1], [0]]
  where(cond, x, y) = [[1, 2], [7, 8]]

  cond = [0, 1]
  where(cond, 1, -1) = [-1, 1]

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
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* param = attrs.as<SqueezeAttrs>();
  ICHECK(param != nullptr);
  std::vector<IndexExpr> result_shape;
  // if axes is None, squeeze all axes of dimension 1
  if (!param->axis.defined()) {
    for (const auto& e : data->shape) {
      if (!e.as<IntImmNode>()) {
        LOG(FATAL) << "axis needs to be defined for dynamic input.";
      }
      const int64_t* axis_ptr = tir::as_const_int(e);
      ICHECK(axis_ptr != nullptr) << "the axes attribute must be concrete";
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
      ICHECK_GE(axis_val, 0);
      ICHECK_LT(axis_val, original_shape.size());
      original_shape.at(axis_val).second = false;
    }
    for (const auto& p : original_shape) {
      if (p.second) {
        result_shape.push_back(p.first);
      } else {
        if (const int64_t* axis_ptr = tir::as_const_int(p.first)) {
          ICHECK_EQ(*axis_ptr, 1) << "cannot squeeze axis with dimension not equal to 1";
        }
      }
    }
  }
  reporter->Assign(types[1], TensorType(result_shape, data->dtype));
  return true;
}

Array<te::Tensor> SqueezeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {
  const SqueezeAttrs* param = attrs.as<SqueezeAttrs>();
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 3);
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
  ICHECK(out_ttype != nullptr);
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
  ICHECK_EQ(types.size(), 3);
  const InitOpAttrs* param = attrs.as<InitOpAttrs>();

  const auto* target_shape = types[1].as<TensorTypeNode>();
  DataType out_dtype = types[0].as<TensorTypeNode>()->dtype;

  const IntImmNode* rank = target_shape->shape[0].as<IntImmNode>();
  ICHECK(rank) << "Parameter must have static rank";

  std::vector<IndexExpr> oshape;
  if (param->shape) {
    const Array<Integer>& cshape_array = param->shape.value();
    for (size_t i = 0; i < cshape_array.size(); i++) {
      oshape.push_back(cshape_array[i]);
    }
  } else {
    for (int i = 0; i < rank->value; i++) {
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

bool BroadCastToRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  // types = [data_type, ret_type], broadcast_to_type is in attrs bc static
  ICHECK_EQ(types.size(), 2);

  const InitOpAttrs* param = attrs.as<InitOpAttrs>();
  ICHECK(param);

  DataType out_dtype = types[0].as<TensorTypeNode>()->dtype;
  std::vector<IndexExpr> oshape;

  const Array<Integer>& cshape_array = param->shape.value();
  for (size_t i = 0; i < cshape_array.size(); ++i) {
    oshape.push_back(cshape_array[i]);
  }
  reporter->Assign(types[1], TensorType(oshape, out_dtype));
  return BroadcastRel({types[0], types[1], types[1]}, 2, Attrs(), reporter);
}

Expr MakeBroadCastTo(Expr data, Array<Integer> shape) {
  static const Op& op = Op::Get("broadcast_to");
  auto attrs = make_object<InitOpAttrs>();

  attrs->shape = std::move(shape);
  return Call(op, {data}, Attrs(attrs), {});
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
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(4)
    .add_type_rel("BroadCastTo", BroadCastToRel)
    .set_attr<FTVMCompute>("FTVMCompute", BroadCastToCompute)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

// BroadCastToLike: <A, B> -> B where BroadCast(A, B) = B
bool BroadCastToLikeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
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
  ICHECK(out_ttype != nullptr);
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
    ICHECK(!arr[i].defined() || arr[i].as<IntImmNode>()) << "Expect an int array";
  }
  return Downcast<Array<Integer>>(arr);
}

// strided_slice
TVM_REGISTER_NODE_TYPE(StridedSliceAttrs);

bool StridedSliceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
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
  if (param->begin && param->end && param->strides) {
    // stride will be set as 1 if slice mode is enabled
    std::vector<int64_t> stride_vec(num_axis, 1);
    if (param->slice_mode == "end") {
      for (size_t i = 0; i < param->strides.value().size(); ++i) {
        ICHECK(param->strides.value()[i].defined());
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
        ICHECK_LE(end_v, begin_v) << "strided_slice get empty slice at axis " << i;
        begin_v = std::min(dim_size - 1, begin_v);
        slice_range = begin_v - end_v;
        step = -stride_v;
      } else {
        if (begin_v < 0) begin_v = 0;
        ICHECK_GE(stride_v, 0);
        ICHECK_LE(begin_v, end_v) << "strided_slice get invalid slice at axis " << i;
        end_v = std::min(dim_size, end_v);
        slice_range = end_v - begin_v;
        step = stride_v;
      }
      oshape[i] = tir::make_const(dshape[i].dtype(), (slice_range + step - 1) / step);
    }
  } else {
    ICHECK(param->begin) << "strided_slice recieved invalid begin " << param->begin;
    ICHECK(param->end) << "strided_slice recieved invalid end " << param->end;
    ICHECK(param->strides) << "strided_slice recieved invalid strides " << param->strides;
  }
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Array<Array<Layout>> StridedSliceInferCorrectLayout(const Attrs& attrs,
                                                    const Array<Layout>& new_in_layouts,
                                                    const Array<Layout>& old_in_layouts,
                                                    const Array<tvm::relay::Type>& old_in_types) {
  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    ICHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }

  ICHECK(old_in_layouts.defined());
  ICHECK_GE(old_in_layouts.size(), 1);
  ICHECK(old_in_shapes.defined());
  ICHECK_GE(old_in_shapes.size(), 1);

  auto layout = old_in_layouts[0];
  if (layout.defined() && new_in_layouts.defined()) {
    ICHECK_GE(new_in_layouts.size(), 1);
    auto new_layout = new_in_layouts[0];
    auto shape = old_in_shapes[0];

    // NOTE: Discard "const" qualifier here.
    auto* params = const_cast<StridedSliceAttrs*>(attrs.as<StridedSliceAttrs>());
    ICHECK(params != nullptr);
    Array<Integer> begin, end, strides;
    if (params->begin && params->end && params->strides) {
      for (Integer i : params->strides.value()) {
        ICHECK(i.defined());
        strides.push_back(params->slice_mode == "size" ? 1 : i->value);
      }

      for (Integer i : params->begin.value()) {
        ICHECK(i.defined());
        begin.push_back(i->value);
      }
      for (Integer i : params->end.value()) {
        ICHECK(i.defined());
        end.push_back(i->value);
      }
    }

    Array<Integer> new_begin, new_end, new_strides;

    // Handles layout conversion like NHWC -> NCHW
    auto old_layout_name = layout.name();
    auto new_layout_name = new_layout.name();

    if (old_layout_name.rfind(new_layout_name, 0) != 0 &&
        new_layout_name.rfind(old_layout_name, 0) != 0) {
      if (old_layout_name.size() != new_layout_name.size()) {
        // Not support NHW4c -> NCHW
        return {{Layout::Undef()}, {Layout::Undef()}};
      } else {
        for (size_t i = 0; i < new_layout_name.size(); ++i) {
          auto index = layout.IndexOf(new_layout[i]);
          if (index == -1) {
            return {{Layout::Undef()}, {Layout::Undef()}};
          }

          size_t new_index = static_cast<size_t>(index);
          int64_t bg, ed, st;
          if (strides.defined() && new_index < strides.size() && strides[new_index].defined()) {
            st = strides[new_index]->value;
          } else {
            st = 1;
          }
          if (new_index < begin.size() && begin[new_index].defined()) {
            bg = begin[new_index]->value;
          } else {
            bg = 0;
          }
          if (new_index < end.size() && end[new_index].defined()) {
            ed = end[new_index]->value;
          } else {
            ed = shape[new_index].as<IntImmNode>()->value;
          }

          new_begin.push_back(bg);
          new_end.push_back(ed);
          new_strides.push_back(st);
        }
        params->begin = new_begin;
        params->end = new_end;
        params->strides = new_strides;
        layout = new_layout;
      }
    } else {
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
  }
  return {{layout}, {layout}};
}

Array<te::Tensor> StridedSliceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                      const Type& out_type) {
  const StridedSliceAttrs* param = attrs.as<StridedSliceAttrs>();
  ICHECK(param != nullptr);
  Array<Integer> begin, end, strides;
  Array<PrimExpr> begin_expr, end_expr, strides_expr;
  begin = param->begin.value();
  end = param->end.value();
  strides = param->strides.value();
  if (IsDynamic(out_type)) {
    auto input = inputs[0];
    size_t src_tensor_dim = input->shape.size();
    ICHECK(begin.size() == src_tensor_dim)
        << "for dynamic inputs, len(begin) must equal the input dimension";
    Array<IndexExpr> out_shape;
    for (size_t i = 0; i < src_tensor_dim; ++i) {
      out_shape.push_back(tvm::tir::Var("dim"));
    }
    for (size_t i = 0; i < src_tensor_dim; ++i) {
      int64_t begin_i = begin[i]->value;
      if (begin_i < 0) {
        begin_i += topi::detail::GetConstInt(input->shape[i]);
      }
      begin_expr.push_back(tir::make_const(begin[0].dtype(), begin_i));
      strides_expr.push_back(
          tir::make_const((strides.size() != 0 ? strides[0].dtype() : begin[0].dtype()),
                          (i < strides.size() ? strides[i]->value : 1)));
    }
    return Array<te::Tensor>{te::compute(
        out_shape,
        [&](const Array<tir::Var>& indices) {
          Array<PrimExpr> real_indices;
          for (size_t i = 0; i < src_tensor_dim; ++i) {
            real_indices.push_back(indices[i] * strides_expr[i] + begin_expr[i]);
          }
          return input(real_indices);
        },
        std::string{"T_strided_slice_dynamic"}, std::string{topi::kInjective})};
  } else {
    for (size_t i = 0; i < begin.size(); ++i) {
      begin_expr.push_back(begin[i]);
    }
    for (size_t i = 0; i < end.size(); ++i) {
      end_expr.push_back(end[i]);
    }
    for (size_t i = 0; i < strides.size(); ++i) {
      strides_expr.push_back(strides[i]);
    }
  }
  return Array<te::Tensor>{
      topi::strided_slice(inputs[0], begin_expr, end_expr, strides_expr, param->slice_mode)};
}

// Positional relay function to create StridedSlice operator used by frontend FFI.
Expr MakeStridedSlice(Expr data, Array<Integer> begin, Array<Integer> end, Array<Integer> strides,
                      String slice_mode) {
  auto attrs = make_object<StridedSliceAttrs>();
  attrs->begin = std::move(begin);
  attrs->end = std::move(end);
  attrs->strides = std::move(strides);
  attrs->slice_mode = slice_mode;
  static const Op& op = Op::Get("strided_slice");
  return Call(op, {data}, Attrs(attrs), {});
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
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
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
  ICHECK_EQ(types.size(), 6);
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
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  ICHECK_NE(data->shape.size(), 0) << "Input shape cannot be empty";
  const auto param = attrs.as<SplitAttrs>();
  ICHECK(param != nullptr);
  auto axis = param->axis;
  if (axis < 0) {
    axis += data->shape.size();
  }
  ICHECK_LT(axis, data->shape.size()) << "axis should be within the input dimension range.";
  ICHECK_GE(axis, 0) << "axis should be within the input dimension range.";

  if (const IntImmNode* sections = param->indices_or_sections.as<IntImmNode>()) {
    if (!data->shape[axis].as<AnyNode>()) {
      ICHECK(reporter->Assert(indexmod(data->shape[axis], sections->value) ==
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
      ICHECK(reporter->Assert(Downcast<IndexExpr>(indices[i]) > begin))
          << "indices_or_sections need to be a sorted ascending list";
      std::vector<IndexExpr> oshape(data->shape.begin(), data->shape.end());
      oshape[axis] = Downcast<IndexExpr>(indices[i]) - begin;
      begin = Downcast<IndexExpr>(indices[i]);
      auto vec_type = TensorType(oshape, data->dtype);
      fields.push_back(vec_type);
    }
    if (!data->shape[axis].as<AnyNode>()) {
      ICHECK(reporter->Assert(begin < data->shape[axis]))
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
  ICHECK(param != nullptr);

  if (const IntImmNode* sections = param->indices_or_sections.as<IntImmNode>()) {
    int64_t num_sections = sections->value;
    return Array<te::Tensor>{topi::split_sections(inputs[0], num_sections, param->axis)};
  } else {
    Array<PrimExpr> indices;
    for (auto i : Downcast<Array<Integer>>(param->indices_or_sections)) {
      indices.push_back(IntImm(DataType::Int(32), i.as<IntImmNode>()->value));
    }
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
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }

  const auto* target = types[1].as<TensorTypeNode>();
  if (target == nullptr) {
    return false;
  }

  const auto param = attrs.as<SliceLikeAttrs>();
  ICHECK(param != nullptr);

  const Array<IndexExpr>& dshape = data->shape;
  const Array<IndexExpr>& target_shape = target->shape;
  std::vector<IndexExpr> oshape(dshape.begin(), dshape.end());

  if (!param->axes.defined()) {
    for (size_t i = 0; i < dshape.size(); ++i) {
      if (i < target_shape.size()) {
        oshape[i] = target_shape[i];
        ICHECK(reporter->Assert(oshape[i] <= dshape[i]))
            << "End index of axis " << i << " exceeds input shape: " << oshape[i] << " vs "
            << dshape[i];
      }
    }
  } else {
    ICHECK(param->axes.size() != 0) << "Axes cannot be empty.";
    for (Integer val : param->axes) {
      int axis = val->value;
      if (axis < 0) {
        axis += dshape.size();
      }
      ICHECK(axis < static_cast<int>(target_shape.size()))
          << "Axis " << axis << " exceeds dimension " << target_shape.size() << " of target_shape.";
      oshape[axis] = target_shape[axis];
      ICHECK(reporter->Assert(oshape[axis] <= dshape[axis]))
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

Array<Array<Layout>> SliceLikeInferCorrectLayout(const Attrs& attrs,
                                                 const Array<Layout>& new_in_layouts,
                                                 const Array<Layout>& old_in_layouts,
                                                 const Array<tvm::relay::Type>& old_in_types) {
  Array<Integer> new_axes;
  if (old_in_layouts.defined() && new_in_layouts.defined()) {
    ICHECK_EQ(new_in_layouts.size(), 2);
    ICHECK_EQ(new_in_layouts[0]->name, new_in_layouts[1]->name);
    ICHECK_EQ(old_in_layouts.size(), 2);
    ICHECK_EQ(old_in_layouts[0]->name, old_in_layouts[1]->name);

    auto old_layout = old_in_layouts[0];
    auto new_layout = new_in_layouts[0];

    // Discard "const" qualifier.
    auto* params = const_cast<SliceLikeAttrs*>(attrs.as<SliceLikeAttrs>());
    ICHECK(params != nullptr);

    for (auto axis : params->axes) {
      auto new_axis = new_layout.IndexOf(old_layout[axis->value]);
      // Cannot find the target axis in the new layout.
      if (new_axis == -1) {
        new_axes.clear();
        break;
      }
      new_axes.push_back(new_axis);
    }
    if (!new_axes.empty()) {
      params->axes = std::move(new_axes);
      return Array<Array<Layout>>({{new_layout, new_layout}, {new_layout}});
    }
  }

  if (old_in_layouts.defined()) {
    ICHECK_EQ(old_in_layouts.size(), 2);
    return {{old_in_layouts[0], old_in_layouts[1]}, {old_in_layouts[1]}};
  }
  return Array<Array<Layout>>({{Layout::Undef(), Layout::Undef()}, {Layout::Undef()}});
}

Array<te::Tensor> SliceLikeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                   const Type& out_type) {
  const auto* param = attrs.as<SliceLikeAttrs>();
  ICHECK(param != nullptr);
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
        ICHECK_LE(topi::GetConstInt(end_idx[i]), topi::GetConstInt(src_shape[i]))
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
      ICHECK_LE(topi::GetConstInt(end_idx[axis]), topi::GetConstInt(src_shape[axis]))
          << "End index of axis " << axis
          << " exceeds input shape: " << topi::GetConstInt(end_idx[axis]) << " vs "
          << topi::GetConstInt(src_shape[axis]);
    }
  }
  return Array<te::Tensor>{topi::strided_slice(inputs[0], begin_idx, end_idx, strides, "end")};
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
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", SliceLikeInferCorrectLayout)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// relay.layout_transform
TVM_REGISTER_NODE_TYPE(LayoutTransformAttrs);

Array<te::Tensor> LayoutTransformCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                         const Type& out_type) {
  const auto* param = attrs.as<LayoutTransformAttrs>();
  ICHECK(param != nullptr);
  return Array<te::Tensor>{topi::layout_transform(inputs[0], param->src_layout, param->dst_layout)};
}

bool LayoutTransformRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "LayoutTransform: expect input data type to be TensorType but get " << types[0];
    return false;
  }
  const LayoutTransformAttrs* params = attrs.as<LayoutTransformAttrs>();

  Layout src_layout(params->src_layout);
  Layout dst_layout(params->dst_layout);

  ICHECK(src_layout.defined() && dst_layout.defined()) << "cannot convert from/to undefined layout";
  auto layout_converter = tir::BijectiveLayout(src_layout, dst_layout);
  ICHECK(layout_converter.defined())
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

// relay.auto_scheduler_layout_transform
TVM_REGISTER_NODE_TYPE(AutoSchedulerLayoutTransformAttrs);

Array<te::Tensor> AutoSchedulerLayoutTransformCompute(const Attrs& attrs,
                                                      const Array<te::Tensor>& inputs,
                                                      const Type& out_type) {
  const auto* param = attrs.as<AutoSchedulerLayoutTransformAttrs>();
  CHECK(param != nullptr);
  return Array<te::Tensor>{
      topi::auto_scheduler_layout_transform(inputs[0], param->src_layout, param->dst_layout)};
}

bool AutoSchedulerLayoutTransformRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                                     const TypeReporter& reporter) {
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  const AutoSchedulerLayoutTransformAttrs* params = attrs.as<AutoSchedulerLayoutTransformAttrs>();

  Array<IndexExpr> dst_shape;
  std::vector<std::string> dst_axes;

  topi::parse_auto_scheduler_layout(params->dst_layout, &dst_shape, &dst_axes);

  reporter->Assign(types[1], TensorType(dst_shape, data->dtype));
  return true;
}

Expr MakeAutoSchedulerLayoutTransform(Expr data, String src_layout, String dst_layout) {
  auto attrs = make_object<AutoSchedulerLayoutTransformAttrs>();
  attrs->src_layout = std::move(src_layout);
  attrs->dst_layout = std::move(dst_layout);
  static const Op& op = Op::Get("auto_scheduler_layout_transform");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.auto_scheduler_layout_transform")
    .set_body_typed(MakeAutoSchedulerLayoutTransform);

RELAY_REGISTER_OP("auto_scheduler_layout_transform")
    .describe(R"code(Transform the input kernel layout.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<AutoSchedulerLayoutTransformAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_type_rel("auto_scheduler_layout_transform", AutoSchedulerLayoutTransformRel)
    .set_support_level(5)
    .set_attr<FTVMCompute>("FTVMCompute", AutoSchedulerLayoutTransformCompute);

// relay._contrib_reverse_reshape
Expr MakeReverseReshape(Expr data, Array<Integer> newshape) {
  auto attrs = make_object<ReshapeAttrs>();
  attrs->newshape = std::move(newshape);
  static const Op& op = Op::Get("contrib_reverse_reshape");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.contrib_reverse_reshape").set_body_typed(MakeReverseReshape);

RELAY_REGISTER_OP("contrib_reverse_reshape")
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
    .add_type_rel("ReverseReshape", ReverseReshapeRel)
    .set_attr<FTVMCompute>("FTVMCompute", ReshapeCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// gather operator
TVM_REGISTER_NODE_TYPE(GatherAttrs);

bool GatherRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // `types` contains: [data, indices, result]
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* indices = types[1].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "Gather: expect input data type to be TensorType but get " << types[0];
    return false;
  }
  if (indices == nullptr) {
    ICHECK(types[1].as<IncompleteTypeNode>())
        << "Gather: expect indices type to be TensorType but get " << types[1];
    return false;
  }
  ICHECK(indices->dtype.is_int()) << "indices of take must be tensor of integer";
  const auto param = attrs.as<GatherAttrs>();
  ICHECK(param != nullptr);
  ICHECK(param->axis.defined());

  const auto ndim_data = data->shape.size();
  const auto ndim_indices = indices->shape.size();
  int axis = param->axis->value;
  ICHECK_EQ(ndim_data, ndim_indices);
  if (axis < 0) {
    axis += ndim_data;
  }
  ICHECK_GE(axis, 0);
  ICHECK_LT(axis, ndim_data);

  std::vector<IndexExpr> oshape;
  oshape.reserve(ndim_data);
  for (size_t i = 0; i < ndim_data; ++i) {
    if (i == (size_t)axis) {
      const int64_t* indice_shape_i = tir::as_const_int(indices->shape[i]);
      ICHECK_GE(*indice_shape_i, 1);
    } else {
      ICHECK(reporter->AssertEQ(indices->shape[i], data->shape[i]));
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
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* indices = types[1].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "GatherND: expect input data type to be TensorType but get " << types[0];
    return false;
  }
  if (indices == nullptr) {
    ICHECK(types[1].as<IncompleteTypeNode>())
        << "GatherND: expect indices type to be TensorType but get " << types[1];
    return false;
  }
  const size_t ndim = data->shape.size();
  const IntImmNode* mdim = indices->shape[0].as<IntImmNode>();
  const size_t kdim = indices->shape.size() - 1;
  ICHECK(size_t(mdim->value) <= ndim) << "GatherND: indices shape does satisfy.";

  Array<IndexExpr> oshape;
  for (size_t i = 1; i < kdim + 1; ++i) oshape.push_back(indices->shape[i]);
  for (size_t i = mdim->value; i < ndim; ++i) oshape.push_back(data->shape[i]);
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
    .add_argument("indices", "Tensor", "The indices of values to gather.")
    .set_support_level(3)
    .add_type_rel("GatherND", GatherNDRel)
    .set_attr<FTVMCompute>("FTVMCompute", GatherNDCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// relay.sequence_mask
TVM_REGISTER_NODE_TYPE(SequenceMaskAttrs);

bool SequenceMaskRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  // `types` contains: [data, valid_length, result]
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* valid_length = types[1].as<TensorTypeNode>();
  ICHECK(data);
  ICHECK(valid_length);
  const auto param = attrs.as<SequenceMaskAttrs>();
  Array<IndexExpr> valid_length_shape;
  ICHECK(param->axis == 0 || param->axis == 1);
  valid_length_shape.push_back(data->shape[1 - param->axis]);
  reporter->Assign(types[1], TensorType(valid_length_shape, valid_length->dtype));
  reporter->Assign(types[2], types[0]);
  return true;
}

Array<te::Tensor> SequenceMaskCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                      const Type& out_type) {
  const auto* param = attrs.as<SequenceMaskAttrs>();
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 4);
  const auto* indices = types[0].as<TensorTypeNode>();
  ICHECK(indices);

  const auto param = attrs.as<OneHotAttrs>();
  ICHECK_GT(param->depth, 0);

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
  ICHECK(param != nullptr);
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
  ICHECK_EQ(types.size(), 3);

  const auto* indices = types[0].as<TensorTypeNode>();
  if (indices == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "unravel_index: expect input type to be TensorType but get " << types[0];
    return false;
  }
  ICHECK(indices->dtype.is_int()) << "indices of unravel_index must be tensor of integer";

  const auto* shape = types[1].as<TensorTypeNode>();
  if (shape == nullptr) {
    ICHECK(types[1].as<IncompleteTypeNode>())
        << "unravel_index: expect input type to be TensorType but get " << types[1];
    return false;
  }
  ICHECK(indices->dtype.is_int()) << "shape of unravel_index must be tensor of integer";

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
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Tensor", "The shape tensor.")
    .set_support_level(3)
    .add_type_rel("UnRavelIndexRel", UnRavelIndexRel)
    .set_attr<FTVMCompute>("FTVMCompute", UnRavelIndexCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// sparse_to_dense
TVM_REGISTER_NODE_TYPE(SparseToDenseAttrs);

bool SparseToDenseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(num_inputs, 3);
  auto sparse_indices = types[0].as<TensorTypeNode>();
  auto sparse_values = types[1].as<TensorTypeNode>();
  auto default_value = types[2].as<TensorTypeNode>();

  if (sparse_indices == nullptr || sparse_values == nullptr || default_value == nullptr) {
    return false;
  }

  ICHECK(sparse_indices->dtype.is_int()) << "sparse_indices must be tensor of integers";

  ICHECK_LE(sparse_indices->shape.size(), 3)
      << "sparse_indices must be a tensor of either 0D, 1D or 2D";

  ICHECK_LE(sparse_values->shape.size(), 2) << "sparse_values must be a tensor of either 0D, 1D";

  ICHECK_EQ(default_value->shape.size(), 0) << "default_value should be a scalar";

  const auto* param = attrs.as<SparseToDenseAttrs>();
  ICHECK(param != nullptr);

  Array<IndexExpr> oshape;
  for (auto i : param->output_shape) {
    oshape.push_back(i);
  }
  reporter->Assign(types[3], TensorType(oshape, sparse_values->dtype));
  return true;
}

Array<te::Tensor> SparseToDenseCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                       const Type& out_type) {
  ICHECK_EQ(inputs.size(), 3);
  const auto* param = attrs.as<SparseToDenseAttrs>();
  ICHECK(param != nullptr);
  Array<IndexExpr> output_shape;
  for (auto val : param->output_shape) {
    output_shape.push_back(val);
  }
  return {topi::sparse_to_dense(inputs[0], output_shape, inputs[1], inputs[2]())};
}

Expr MakeSparseToDense(Expr indices, Array<Integer> output_shape, Expr values, Expr default_value) {
  auto attrs = make_object<SparseToDenseAttrs>();
  attrs->output_shape = std::move(output_shape);
  static const Op& op = Op::Get("sparse_to_dense");
  return Call(op, {indices, values, default_value}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relay.op._make.sparse_to_dense").set_body_typed(MakeSparseToDense);

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

// relay.matrix_set_diag
TVM_REGISTER_NODE_TYPE(MatrixSetDiagAttrs);

bool MatrixSetDiagRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  // `types` contains: [input, diagonal, result]
  ICHECK_EQ(types.size(), 3);

  const auto* input = types[0].as<TensorTypeNode>();
  ICHECK(input);

  const auto* diagonal = types[1].as<TensorTypeNode>();
  ICHECK(diagonal);

  const auto param = attrs.as<MatrixSetDiagAttrs>();
  ICHECK_GE(param->k2, param->k1);

  int d_ndims = diagonal->shape.size();
  int i_ndims = input->shape.size();

  reporter->Assert(input->shape[i_ndims - 2] > -param->k1);
  reporter->Assert(input->shape[i_ndims - 1] > param->k2);

  for (int i = 0; i < d_ndims - 2; i++) {
    reporter->AssertEQ(input->shape[i], diagonal->shape[i]);
  }
  if (param->k1 != param->k2) {
    reporter->AssertEQ(diagonal->shape[d_ndims - 2], param->k2 - param->k1 + 1);
  } else if (d_ndims >= 2) {
    reporter->AssertEQ(input->shape[d_ndims - 2], diagonal->shape[d_ndims - 2]);
  }
  auto max_diag_len = if_then_else(input->shape[i_ndims - 2] + (param->k2 > 0 ? param->k2 : 0) <=
                                       input->shape[i_ndims - 1] + (param->k1 < 0 ? -param->k1 : 0),
                                   input->shape[i_ndims - 2] + (param->k2 > 0 ? param->k2 : 0),
                                   input->shape[i_ndims - 1] + (param->k1 < 0 ? -param->k1 : 0));
  reporter->AssertEQ(diagonal->shape[d_ndims - 1], max_diag_len);

  reporter->Assign(types[2], TensorType(input->shape, input->dtype));
  return true;
}

Array<te::Tensor> MatrixSetDiagCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                       const Type& out_type) {
  const auto* param = attrs.as<MatrixSetDiagAttrs>();
  ICHECK(param != nullptr);
  return Array<te::Tensor>{topi::matrix_set_diag(inputs[0], inputs[1], param->k1, param->k2,
                                                 param->super_diag_right_align,
                                                 param->sub_diag_right_align)};
}

Expr MakeMatrixSetDiag(Expr input, Expr diagonal, int k1, int k2, bool super_diag_right_align,
                       bool sub_diag_right_align) {
  auto attrs = make_object<MatrixSetDiagAttrs>();
  attrs->k1 = k1;
  attrs->k2 = k2;
  attrs->super_diag_right_align = super_diag_right_align;
  attrs->sub_diag_right_align = sub_diag_right_align;
  static const Op& op = Op::Get("matrix_set_diag");
  return Call(op, {input, diagonal}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.matrix_set_diag").set_body_typed(MakeMatrixSetDiag);

RELAY_REGISTER_OP("matrix_set_diag")
    .describe(
        R"code(Returns a tensor with the diagonals of input tensor replaced with the provided diagonal values.
        **input** Input tensor.
        **diagonal** Values to be filled in the diagonal.
        **k1** Lower limit (included) of the range of diagonals.
        **k2** Upper limit (included) of the range of diagonals.
        **super_diag_right_align** Bool, true iff super-diagonal is right aligned (left-padded).
        **sub_diag_right_align** Bool, true iff sub-diagonal is right aligned (left-padded).
    )code" TVM_ADD_FILELINE)
    .set_attrs_type<MatrixSetDiagAttrs>()
    .set_num_inputs(2)
    .add_argument("input", "Tensor", "Input Tensor.")
    .add_argument("diagonal", "Tensor", "Values to be filled in the diagonal.")
    .set_support_level(10)
    .add_type_rel("MatrixSetDiag", MatrixSetDiagRel)
    .set_attr<FTVMCompute>("FTVMCompute", MatrixSetDiagCompute)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// adv_index
bool AdvIndexRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  ICHECK_EQ(num_inputs, 1);
  auto inputs = types[0].as<TupleTypeNode>();
  auto data = inputs->fields[0].as<TensorTypeNode>();

  if (inputs == nullptr || data == nullptr) {
    return false;
  }

  Array<IndexExpr> oshape;
  Array<IndexExpr> broadcast_shape;
  int64_t num_picked_elems = 1;

  if (inputs->fields.size() == 2) {
    broadcast_shape = inputs->fields[1].as<TensorTypeNode>()->shape;
  } else {
    for (size_t i = 1; i < inputs->fields.size(); ++i) {
      auto index_type = inputs->fields[i].as<TensorTypeNode>();
      if (index_type == nullptr) {
        return false;
      }
      ICHECK(index_type->dtype.is_int()) << "indices must be tensor of integers";

      int64_t flatten_len = 1;
      bool has_dyn_shape = false;
      for (const auto& dim : index_type->shape) {
        const IntImmNode* axis_len = dim.as<IntImmNode>();
        if (!axis_len) {
          // If dynamic shape appears, just use the first shape
          broadcast_shape = index_type->shape;
          has_dyn_shape = true;
          break;
        }
        flatten_len *= axis_len->value;
      }
      if (has_dyn_shape) break;
      if (flatten_len > num_picked_elems) {
        num_picked_elems = flatten_len;
        broadcast_shape = index_type->shape;
      }
    }
  }

  for (const auto& dim : broadcast_shape) {
    oshape.push_back(dim);
  }
  for (size_t i = inputs->fields.size() - 1; i < data->shape.size(); ++i) {
    oshape.push_back(data->shape[i]);
  }
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> AdvIndexCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                  const Type& out_type) {
  Array<te::Tensor> indices;
  for (size_t i = 1; i < inputs.size(); ++i) {
    indices.push_back(inputs[i]);
  }
  return {topi::adv_index(inputs[0], indices)};
}

Expr MakeAdvIndex(Expr inputs) {
  static const Op& op = Op::Get("adv_index");
  return Call(op, {inputs}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.adv_index").set_body_typed(MakeAdvIndex);

RELAY_REGISTER_OP("adv_index")
    .describe(R"code(Numpy style advanced indexing. Index with a list of tensors.
    )code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_support_level(3)
    .add_argument("inputs", "Tuple of Tensors", "Input tensor and indices.")
    .add_type_rel("AdvIndex", AdvIndexRel)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_attr<FTVMCompute>("FTVMCompute", AdvIndexCompute);

TVM_REGISTER_NODE_TYPE(ScanopAttrs);

bool ScanopRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 2) << "Expects two types, one for the input and another for the output";
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "Scanop: expect input type to be TensorType but get " << types[0];
    return false;
  }

  const auto* param = attrs.as<ScanopAttrs>();

  auto dtype = param->dtype;
  if (dtype.is_void()) {
    dtype = data->dtype;
  }

  if (param->axis.defined()) {
    reporter->Assign(types[1], TensorType(data->shape, dtype));
  } else {
    auto prod = data->shape[0];
    for (size_t i = 1; i < data->shape.size(); ++i) {
      prod = prod * data->shape[i];
    }
    reporter->Assign(types[1], TensorType({prod}, dtype));
  }

  return true;
}

Expr MakeCumsum(Expr data, Integer axis, DataType dtype, Bool exclusive) {
  auto attrs = make_object<ScanopAttrs>();
  attrs->dtype = dtype;
  attrs->axis = axis;
  attrs->exclusive = exclusive;
  static const Op& op = Op::Get("cumsum");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.cumsum").set_body_typed(MakeCumsum);

RELAY_REGISTER_OP("cumsum")
    .describe(
        R"doc(Return the cumulative sum of the elements along a given axis.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Cumsum", ScanopRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

Expr MakeCumprod(Expr data, Integer axis, DataType dtype, Bool exclusive) {
  auto attrs = make_object<ScanopAttrs>();
  attrs->dtype = dtype;
  attrs->axis = axis;
  attrs->exclusive = exclusive;
  static const Op& op = Op::Get("cumprod");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.cumprod").set_body_typed(MakeCumprod);

RELAY_REGISTER_OP("cumprod")
    .describe(
        R"doc(Return the cumulative product of the elements along a given axis.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Cumprod", ScanopRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_NODE_TYPE(UniqueAttrs);

bool UniqueRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types: [data, result]
  ICHECK_EQ(types.size(), 2) << "Unique: expect 2 types but " << types.size() << " provided";
  ICHECK_EQ(num_inputs, 1) << "Unique: expect 1 inputs but " << num_inputs << " provided";
  auto data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "Unique: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const int ndim = static_cast<int>(data->shape.size());
  ICHECK_EQ(ndim, 1) << "Unique: input must be 1-D tensor";
  ICHECK_EQ(data->dtype.is_int(), true) << "Unique: input must have int32 or int64 dtype";
  std::vector<Type> fields;
  fields.push_back(TensorType(data->shape, data->dtype));               // unique
  fields.push_back(TensorType(data->shape, DataType::Int(32)));         // indices
  fields.push_back(TensorType(Array<PrimExpr>{1}, DataType::Int(32)));  // num_unique
  const auto* param = attrs.as<UniqueAttrs>();
  if (param->return_counts) {
    fields.push_back(TensorType(data->shape, DataType::Int(32)));  // counts
  }
  reporter->Assign(types[1], TupleType(Array<Type>(fields)));
  return true;
}

Expr MakeUnique(Expr data, bool sorted, bool return_counts) {
  auto attrs = make_object<UniqueAttrs>();
  attrs->sorted = sorted;
  attrs->return_counts = return_counts;
  static const Op& op = Op::Get("unique");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.unique").set_body_typed(MakeUnique);

RELAY_REGISTER_OP("unique")
    .describe(
        R"code(This operation returns the unique elements and the new index of each item in a given 1-D array.
    )code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .add_type_rel("unique", UniqueRel)
    .set_support_level(3)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
}  // namespace relay
}  // namespace tvm
