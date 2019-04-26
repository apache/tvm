/*!
 *  Copyright (c) 2018 by Contributors
 * \file nms.cc
 * \brief Non-maximum suppression operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/algorithm.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ArgsortAttrs);

bool ArgsortRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  // `types` contains: [data, result]
  const ArgsortAttrs* param = attrs.as<ArgsortAttrs>();
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "Argsort: expect input type to be TensorType but get "
        << types[0];
    return false;
  }
  CHECK_EQ(param->dtype, Float(32));
  reporter->Assign(types[1], TensorTypeNode::make(data->shape, param->dtype));
  return true;
}

Expr MakeArgsort(Expr data,
                 int axis,
                 bool is_ascend,
                 DataType dtype) {
  auto attrs = make_node<ArgsortAttrs>();
  attrs->axis = axis;
  attrs->is_ascend = is_ascend;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("argsort");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op._make.argsort")
.set_body_typed(MakeArgsort);

RELAY_REGISTER_OP("argsort")
.describe(R"doc(Returns the indices that would sort an
input array along the given axis.
)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.ArgsortAttrs")
.add_argument("data", "Tensor", "Input data.")
.set_support_level(6)
.add_type_rel("Argsort", ArgsortRel);
}  // namespace relay
}  // namespace tvm
