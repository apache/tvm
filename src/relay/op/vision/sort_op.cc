/*!
 *  Copyright (c) 2018 by Contributors
 * \file nms.cc
 * \brief Non-maximum suppression operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/vision.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ArgsortAttrs);

bool ArgsortRel(const Array<Type>& types,
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
  reporter->Assign(types[1], TensorTypeNode::make(data->shape, Int(32)));
  return true;
}

Expr MakeArgsort(Expr data,
                 int axis,
                 bool is_ascend) {
  auto attrs = make_node<ArgsortAttrs>();
  attrs->axis = axis;
  attrs->is_ascend = is_ascend;
  static const Op& op = Op::Get("argsort");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op._make.argsort")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
  runtime::detail::unpack_call<Expr, 3>(MakeArgsort, args, rv);
});


RELAY_REGISTER_OP("Argsort")
.describe(R"doc(Returns the indics that would sort an
input array along the given axis.
)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.ArgsortAttrs")
.add_argument("data", "Tensor", "Input data.")
.set_support_level(5)
.add_type_rel("Argsort", ArgsortRel);
}  // namespace relay
}  // namespace tvm
