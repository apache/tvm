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
.describe(R"code(Returns the log of input array, computed element-wise.

.. math::
   log(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("exp")
.describe(R"code(Returns the exp of input array, computed element-wise.

.. math::
   \exp(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);


RELAY_REGISTER_UNARY_OP("sqrt")
.describe(R"code(Returns the sqrt input array, computed element-wise.

.. math::
   sqrt(x)

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

RELAY_REGISTER_UNARY_OP("floor")
.describe(R"code(Returns the floor of input array, computed element-wise.
)code" TVM_ADD_FILELINE)
.set_support_level(3)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("ceil")
.describe(R"code(Returns the ceil of input array, computed element-wise.

.. math::
   ceil(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("trunc")
.describe(R"code(Returns the trunc of input array, computed element-wise.

.. math::
   trunc(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("round")
.describe(R"code(Returns the round of input array, computed element-wise.

.. math::
   round(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("abs")
.describe(R"code(Returns the abs of input array, computed element-wise.

.. math::
   abs(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("tanh")
.describe(R"code(Returns the tanh of input array, computed element-wise.

.. math::
   Y = sinh(X) / cosh(X)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.add_type_rel("Identity", IdentityRel);

RELAY_REGISTER_UNARY_OP("negative")
.describe(R"code(Returns the numeric negative of input array, computed element-wise.

.. math::
   -(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.add_type_rel("Identity", IdentityRel);


}  // namespace relay
}  // namespace tvm
