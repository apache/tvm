/*!
 *  Copyright (c) 2018 by Contributors
 * \file unary.cc
 * \brief Unary operators.
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
.add_type_rel("Identity", IdentityRel);

// data : Tensor[shape, dtype]
// result: Tensor[shape, dtype]


RELAY_REGISTER_UNARY_OP("exp")
.describe(R"code(Returns the exp input array, computed element-wise.

.. math::
   \exp(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);


RELAY_REGISTER_UNARY_OP("sqrt")
.describe(R"code(Returns the sqrt input array, computed element-wise.
)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("zeros_like")
.describe(R"code(Returns an array of zeros, with same type and shape as the input.
)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("ones_like")
.describe(R"code(Returns an array of ones, with same type and shape as the input.
)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("sigmoid")
.describe(R"code(Returns the sigmoid input array, computed element-wise.

.. math::
   sigmoid(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("copy")
.describe(R"code(Copy a tensor.
)code" TVM_ADD_FILELINE)
.set_support_level(3)
.add_type_rel("Identity", IdentityRel);

}  // namespace relay
}  // namespace tvm
