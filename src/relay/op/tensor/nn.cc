/*!
 *  Copyright (c) 2018 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include "../type_relations.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(DenseAttrs);


bool DenseRel(const Array<Type>& types,
              int num_inputs,
              const Attrs& attrs,
              const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const auto* weight = types[1].as<TensorTypeNode>();
  if (weight == nullptr) return false;

  const DenseAttrs* param = attrs.as<DenseAttrs>();
  CHECK(param != nullptr);

  CHECK(static_cast<int>(data->shape.size()) != 0);
  Array<tvm::Expr> wshape = weight->shape;

  if (param->units != 0) {
    // CHECK_EQ(param->units == wshape[wshape.size() - 1])
  }

  Array<tvm::Expr> oshape = data->shape;
  oshape.Set((oshape.size() - 1), wshape[wshape.size() - 1]);

  // assign output type
  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}


// Positional relay function to create dense operator used by frontend FFI.
Expr MakeDense(Expr data,
               Expr weight,
               int units) {
  auto attrs = make_node<DenseAttrs>();
  attrs->units = units;
  static const Op& op = Op::Get("nn.dense");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.dense")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 3>(MakeDense, args, rv);
  });

RELAY_REGISTER_OP("nn.dense")
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight", "2D Tensor", "Weight matrix.")
.set_support_level(2)
.add_type_rel("Dense", DenseRel);


// Positional relay function to create leaky relu operator used by frontend FFI.
Expr MakeLeakyRelu(Expr data,
                   double alpha) {
  auto attrs = make_node<LeakyReluAttrs>();
  attrs->alpha = alpha;
  static const Op& op = Op::Get("nn.leaky_relu");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.leaky_relu")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 2>(MakeLeakyRelu, args, rv);
  });

RELAY_REGISTER_OP("nn.leaky_relu")
.describe(R"code(Leaky version of a Rectified Linear Unit.

`y = x > 0 ? x : alpha * x`

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "Input data.")
.set_support_level(3)
.add_type_rel("Identity", IdentityRel);

}  // namespace relay
}  // namespace tvm
