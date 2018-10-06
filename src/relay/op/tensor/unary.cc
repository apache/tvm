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

// Concat
TVM_REGISTER_API("relay.op._make.concat")
  .set_body_typed<Expr(Expr)>([](Expr tuple) {
      static const Op& op = Op::Get("concat");
    return CallNode::make(op, { tuple }, Attrs(), {});
  });

RELAY_REGISTER_OP("concat")
.set_num_inputs(1)
.add_argument("tuple", "Tuple", "The tupled tensor arguments.")
.set_support_level(1)
.add_type_rel("Concat", ConcatRel);

// Clip
struct ClipAttrs : public tvm::AttrsNode<ClipAttrs> {
  double a_min;
  double a_max;

  TVM_DECLARE_ATTRS(ClipAttrs, "relay.attrs.ClipAttrs") {
  TVM_ATTR_FIELD(a_min)
    .describe("The minimum clip value.");
  TVM_ATTR_FIELD(a_max)
    .describe("The maximum clip value.");
  }
};

TVM_REGISTER_API("relay.op._make.clip")
  .set_body_typed<Expr(Expr, double, double)>([](Expr a, double a_min, double a_max) {
      auto attrs = make_node<ClipAttrs>();
      attrs->a_min = a_min;
      attrs->a_max = a_max;
      static const Op& op = Op::Get("clip");
    return CallNode::make(op, {a}, Attrs(attrs), {});
  });

RELAY_REGISTER_OP("clip")
  .describe(R"code(Clip tensor values.
  This function takes a tensor, a minimum value `a_min`, and a maximum value `a_max`, and returns a clipped tensor where all values below `a_min` are set to `a_min` and all values above `a_max` are set to `a_max`. `a_min` and `a_max` are cast to the tensor's dtype.
  )code" TVM_ADD_FILELINE)
  .set_num_inputs(1)
  .add_argument("tensor", "Tensor", "The input tensor.")
  .set_support_level(3)
  .add_type_rel("Clip", IdentityRel);

}  // namespace relay
}  // namespace tvm
