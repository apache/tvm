/*!
 *  Copyright (c) 2018 by Contributors
 * \file elemwise.cc
 * \brief Elementwise operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include "../type_relations.h"

namespace tvm {
namespace relay {

// Quick helper macro
// - Expose a positional make function to construct the node.
// - Register op to the registry.
//
// We make the decision to always only expose positional argument.
// We will do rewrapping in the frontend to support language
// sugars such as keyword arguments and default value.
//
#define RELAY_REGISTER_UNARY_OP(OpName)               \
  TVM_REGISTER_API("relay.op._make." OpName)          \
  .set_body_typed<Expr(Expr)>([](Expr data) {         \
      static const Op& op = Op::Get(OpName);          \
    return CallNode::make(op, {data}, Attrs(), {});   \
    });                                               \
  RELAY_REGISTER_OP(OpName)                           \
  .set_num_inputs(1)                                  \
  .add_argument("data", "Tensor", "The input tensor.")


RELAY_REGISTER_UNARY_OP("log")
.describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_func("Log", IdentityRel);

// data : Tensor[shape, dtype]
// result: Tensor[shape, dtype]


RELAY_REGISTER_UNARY_OP("exp")
.describe(R"code(Returns the exp input array, computed element-wise.

.. math::
   \exp(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_func("Exp", IdentityRel);


RELAY_REGISTER_UNARY_OP("sqrt")
.describe(R"code(Returns the sqrt input array, computed element-wise.

.. math::
   sqrt(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_func("Sqrt", IdentityRel);

// Addition
TVM_REGISTER_API("relay.op._make.add")
  .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {
      static const Op& op = Op::Get("add");
    return CallNode::make(op, {lhs, rhs}, Attrs(), {});
  });

RELAY_REGISTER_OP("add")
  .set_num_inputs(2)
  .add_argument("lhs", "Tensor", "The left hand side tensor.")
  .add_argument("rhs", "Tensor", "The right hand side tensor.")
  .set_support_level(1)
  .add_type_func("Broadcast", BroadcastRel);

  // def broadcast(s1, s2):
  // ...
  //
  // input1: Tensor[dtype, s1]
  // input2: Tensor[dtype, s2]
  // output: Tensor[dtype, broadcast(s1, s2)]

// Addition
TVM_REGISTER_API("relay.op._make.subtract")
  .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {
      static const Op& op = Op::Get("subtract");
    return CallNode::make(op, {lhs, rhs}, Attrs(), {});
  });

RELAY_REGISTER_OP("subtract")
  .set_num_inputs(2)
  .add_argument("lhs", "Tensor", "The left hand side tensor.")
  .add_argument("rhs", "Tensor", "The right hand side tensor.")
  .set_support_level(1)
  .add_type_func("BroadcastComp", BroadcastCompRel);

  // def broadcast(s1, s2):
  // ...
  //
  // input1: Tensor[dtype, s1]
  // input2: Tensor[dtype, s2]
  // output: Tensor[dtype, broadcast(s1, s2)]

// Addition
TVM_REGISTER_API("relay.op._make.equal")
  .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {
      static const Op& op = Op::Get("equal");
    return CallNode::make(op, {lhs, rhs}, Attrs(), {});
  });

RELAY_REGISTER_OP("equal")
  .set_num_inputs(2)
  .add_argument("lhs", "Tensor", "The left hand side tensor.")
  .add_argument("rhs", "Tensor", "The right hand side tensor.")
  .set_support_level(1)
  .add_type_func("BroadcastComp", BroadcastCompRel);

}  // namespace relayv
}  // namespace tvm
