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
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(1)
.add_type_rel("ExpandDims", ExpandDimsRel);

/* relay.concatenate */

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

}  // namespace relay
}  // namespace tvm
