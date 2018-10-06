/*!
 *  Copyright (c) 2018 by Contributors
 * \file transform.cc
 * \brief Transform operators.
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/transform.h>
#include <vector>


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
  const ExpandDimsAttrs* param = attrs.as<ExpandDimsAttrs>();
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
  const ConcatenateAttrs* param = attrs.as<ConcatenateAttrs>();
  const auto& first = Downcast<TensorType>(tensor_tuple->fields[0]);
  // Sanity check: ndim and dtype.
  const int ndim = static_cast<int>(first->shape.size());
  const DataType dtype = first->dtype;
  for (const Type& ele: tensor_tuple->fields) {
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
  std::vector<IndexExpr> oshape; // weird that I cannot directly construct.
  oshape.reserve(ndim);
  for (const IndexExpr& dim: first->shape) {
    oshape.push_back(dim);
  }
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


}  // namespace relay
}  // namespace tvm
