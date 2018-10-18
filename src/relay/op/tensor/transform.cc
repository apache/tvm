/*!
 *  Copyright (c) 2018 by Contributors
 * \file transform.cc
 * \brief Transform operators.
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/ir_operator.h>
#include <vector>
#include "../op_common.h"


namespace tvm {
namespace relay {

/* relay.expand_dims */

TVM_REGISTER_NODE_TYPE(ExpandDimsAttrs);

bool ExpandDimsRel(const Array<Type>& types,
                   int num_inputs,
                   const Attrs& attrs,
                   const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
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
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 3>(MakeExpandDims, args, rv);
});

RELAY_REGISTER_OP("expand_dims")
.describe(R"code(Insert `num_newaxis` axises at the position given by `axis`

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.ExpandDimsAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(1)
.add_type_rel("ExpandDims", ExpandDimsRel);

TVM_REGISTER_NODE_TYPE(ConcatenateAttrs);

bool ConcatenateRel(const Array<Type>& types,
                    int num_inputs,
                    const Attrs& attrs,
                    const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
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

Expr MakeConcatenate(Expr data,
                     int axis) {
  auto attrs = make_node<ConcatenateAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("concatenate");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.concatenate")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeConcatenate, args, rv);
});

RELAY_REGISTER_OP("concatenate")
.describe(R"code(Concatenate the input tensors along the given axis.

- **data** : A list of tensors.

- **axis** : The axis along which the tensors are concatenated.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input list of tensors.")
.set_support_level(1)
.add_type_rel("Concatenate", ConcatenateRel);

/* relay.transpose */

bool TransposeRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* param = attrs.as<TransposeAttrs>();
  const int ndim = data->shape.size();
  const Array<IndexExpr>& axes = param->axes;
  // check dimension match
  CHECK(axes.empty() || static_cast<int>(axes.size()) == ndim)
    << "Dimension mismatch: axes has " << axes.size() << " elements"
    << ", but data.ndim = " << ndim;
  // construct int_axes
  std::vector<int> int_axes;
  int_axes.reserve(ndim);
  if (axes.empty()) {
    for (int i = ndim - 1; i >= 0; --i) {
      int_axes.push_back(i);
    }
  } else {
    std::vector<int> axis_used(ndim, 0);
    for (const IndexExpr& e : axes) {
      const int64_t *axis_ptr = as_const_int(e);
      CHECK(axis_ptr != nullptr);
      int axis = *axis_ptr;
      // sanity check for axis and ndim
      CHECK(-ndim <= axis && axis < ndim)
        << "transpose only allows each `axis` in `axes` in range [-data.ndim, data.ndim)"
        << ", but got axis = " << axis
        << ", and data.ndim = " << ndim;
      axis = axis < 0 ? axis + ndim : axis;
      // sanity check for duplication
      CHECK(!axis_used[axis]) << "Duplicate axes in transpose: " << axis;
      axis_used[axis] = 1;
      int_axes.push_back(axis);
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

Expr MakeTranspose(Expr data,
                   Array<IndexExpr> axes) {
  auto attrs = make_node<TransposeAttrs>();
  attrs->axes = std::move(axes);
  static const Op& op = Op::Get("transpose");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.transpose")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeTranspose, args, rv);
});

RELAY_REGISTER_OP("transpose")
.describe(R"code(Permutes the dimensions of an array.

- **data**: The input data to the operator.

- **axes**: The target axes order, reverse order if not specified.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Transpose", TransposeRel);

/* relay.reshape */

bool ReshapeRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* param = attrs.as<ReshapeAttrs>();
  reporter->Assign(types[1], TensorTypeNode::make(param->newshape, data->dtype));
  return true;
}

Expr MakeReshape(Expr data,
                 Array<IndexExpr> newshape) {
  auto attrs = make_node<ReshapeAttrs>();
  attrs->newshape = std::move(newshape);
  static const Op& op = Op::Get("reshape");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.reshape")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeReshape, args, rv);
});

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
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Reshape", ReshapeRel);

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
  auto axis = (*as_const_int(param->axis));
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

Expr MakeTake(Expr data,
              Expr indices,
              IndexExpr axis) {
  auto attrs = make_node<TakeAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("take");
  return CallNode::make(op, {data, indices}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.take")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 3>(MakeTake, args, rv);
});

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
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("indices", "Tensor", "The indices tensor.")
.set_support_level(2)
.add_type_rel("Take", TakeRel);

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
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 3>(MakeFull, args, rv);
});

RELAY_REGISTER_OP("full")
.describe(R"code(Fill array with scalar value.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("fill_value", "double", "The value to fill.")
.set_support_level(3)
.add_type_rel("Full", FullRel);

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
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeZeros, args, rv);
  });

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
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeOnes, args, rv);
  });

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

Expr MakeFullLike(Expr data,
                  Expr fill_value) {
  static const Op& op = Op::Get("full_like");
  return CallNode::make(op, {data, fill_value}, Attrs(), {});
}

TVM_REGISTER_API("relay.op._make.full_like")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeFullLike, args, rv);
  });

RELAY_REGISTER_OP("full_like")
.describe(R"code(Return an scalar value array with the same shape
and type as the input array.

)code" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("fill_value", "double", "Scalar value to fill.")
.set_support_level(3)
.add_type_rel("FullLike", FullLikeRel);

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

TVM_REGISTER_API("relay.op._make.where")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
  runtime::detail::unpack_call<Expr, 3>(MakeWhere, args, rv);
});

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
.add_type_rel("Where", WhereRel);

Expr MakeSqueeze(Expr data,
                 Array<IndexExpr> axes) {
  auto attrs = make_node<SqueezeAttrs>();
  attrs->axes = std::move(axes);
  static const Op& op = Op::Get("squeeze");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op._make.squeeze")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeSqueeze, args, rv);
  });

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
  // if axes is empty, squeeze all axes of dimension 1
  if (param->axes.size() == 0) {
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
    for (const auto& e : param->axes) {
      const int64_t* axis_ptr = as_const_int(e);
      CHECK(axis_ptr != nullptr);
      original_shape.at(*axis_ptr).second = false;
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

RELAY_REGISTER_OP("squeeze")
.describe(R"code(Squeeze the input tensor at the dimensions given by axes

- **data**: The input data to the operator.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.SqueezeAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Squeeze", SqueezeRel);

}  // namespace relay
}  // namespace tvm
