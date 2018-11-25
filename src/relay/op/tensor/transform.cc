/*!
 *  Copyright (c) 2018 by Contributors
 * \file transform.cc
 * \brief Transform operators.
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/ir_operator.h>
#include <tvm/ir.h>
#include <topi/transform.h>
#include <vector>
#include "../op_common.h"


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

Expr MakeCast(Expr data,
              DataType dtype) {
  auto attrs = make_node<CastAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("cast");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay._make.dtype_cast")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeCast, args, rv);
});

RELAY_REGISTER_OP("cast")
.describe(R"code(Cast the data into a new data type.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.CastAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Cast", CastRel);


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
.set_attrs_type_key("relay.attrs.ConcatenateAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input list of tensors.")
.set_support_level(1)
.add_type_rel("Concatenate", ConcatenateRel);

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

Expr MakeTranspose(Expr data,
                   Array<Integer> axes) {
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
.set_attrs_type_key("relay.attrs.TransposeAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Transpose", TransposeRel);

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
.set_attrs_type_key("relay.attrs.ReshapeAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(3)
.add_type_rel("Reshape", ReshapeRel)
.set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs,
                                         const Array<Tensor>& inputs,
                                         const Type& out_type,
                                         const Target& target) {
  const auto* param = attrs.as<ReshapeAttrs>();
  CHECK(param != nullptr);
  return Array<Tensor>{ topi::reshape(inputs[0], param->newshape) };
});


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
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeReshapeLike, args, rv);
});


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
.set_attr<FTVMCompute>("FTVMCompute", [](const Attrs& attrs,
                                         const Array<Tensor>& inputs,
                                         const Type& out_type,
                                         const Target& target) {
  return Array<Tensor>{ topi::reshape(inputs[0], inputs[1]->shape) };
});


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

Expr MakeTake(Expr data,
              Expr indices,
              Integer axis) {
  auto attrs = make_node<TakeAttrs>();
  attrs->axis = std::move(axis);
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
.set_attrs_type_key("relay.attrs.TakeAttrs")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("indices", "Tensor", "The indices tensor.")
.set_support_level(2)
.add_type_rel("Take", TakeRel);

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
.set_attrs_type_key("relay.attrs.InitOpAttrs")
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
      original_shape.at(e->value).second = false;
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

TVM_REGISTER_API("relay.op._make.collapse_sum_like")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeCollapseSumLike, args, rv);
  });

RELAY_REGISTER_OP("collapse_sum_like")
.describe(R"code(Collapse the first input to match the shape of the second input.
)code" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("collapse_type", "Tensor", "Provide the type to collapse to.")
.set_support_level(10)
.add_type_rel("CollapseSumLike", CollapseSumLikeRel);

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

TVM_REGISTER_API("relay.op._make.broadcast_to_like")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeBroadCastToLike, args, rv);
  });

RELAY_REGISTER_OP("broadcast_to_like")
.describe(R"code(Broadcast the first input to match the shape of the second input.
)code" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("broadcast_type", "Tensor", "Provide the type to broadcast to.")
.set_support_level(10)
.add_type_rel("BroadCastToLike", BroadCastToLikeRel);


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
  .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 4>(MakeStridedSlice, args, rv);
  });


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
.set_attr<TOpPattern>("TOpPattern", kInjective);


// Split
TVM_REGISTER_NODE_TYPE(SplitAttrs);

bool SplitRel(const Array<Type>& types,
              int num_inputs,
              const Attrs& attrs,
              const TypeReporter& reporter) {
  // `types` contains: [data, result]
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
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
    for (uint i = 0; i < indices.size(); ++i) {
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
.add_type_rel("Split", SplitRel);


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

// Adapter function to make int array.
Array<Integer> GetIntArray(Array<IndexExpr> arr) {
  for (size_t i = 0; i < arr.size(); ++i) {
    CHECK(!arr[i].defined() || arr[i].as<IntImm>())
        << "Expect an int array";
  }
  return Array<Integer>(arr.node_);
}

template<typename AttrType>
Array<Tensor> SliceLikeCompute(const Attrs& attrs,
                               const Array<Tensor>& inputs,
                               const Type& out_type,
                               const Target& target) {
  const auto* param = attrs.as<AttrType>();
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
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 3>(MakeSliceLike, args, rv);
});


RELAY_REGISTER_OP("slice_like")
.describe(R"code(Slice the first input respect to the second input.
)code" TVM_ADD_FILELINE)
  .set_attrs_type_key("relay.attrs.SlicelikeAttrs")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("shape_like", "Tensor", "Shape tensor.")
.set_support_level(10)
.add_type_rel("SliceLike", SliceLikeRel)
.set_attr<FTVMCompute>("FTVMCompute", SliceLikeCompute<SliceLikeAttrs>);

}  // namespace relay
}  // namespace tvm
