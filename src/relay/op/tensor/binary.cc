/*!
 *  Copyright (c) 2018 by Contributors
 * \file binary.cc
 * \brief binary broadcast operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include "../type_relations.h"

namespace tvm {
namespace relay {

#define RELAY_REGISTER_BINARY_OP(OpName)                               \
  TVM_REGISTER_API("relay.op._make." OpName)                           \
  .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {           \
      static const Op& op = Op::Get(OpName);                           \
      return CallNode::make(op, {lhs, rhs}, Attrs(), {});              \
    });                                                                \
  RELAY_REGISTER_OP(OpName)                                            \
  .set_num_inputs(2)                                                   \
  .add_argument("lhs", "Tensor", "The left hand side tensor.")         \
  .add_argument("rhs", "Tensor", "The right hand side tensor.")        \
  .add_type_rel("Broadcast", BroadcastRel)

// Addition
RELAY_REGISTER_BINARY_OP("add")
.describe("Elementwise add with with broadcasting")
.set_support_level(1);

RELAY_REGISTER_BINARY_OP("subtract")
.describe("Elementwise substract with broadcasting")
.set_support_level(1);

RELAY_REGISTER_BINARY_OP("right_shift")
.describe("Elementwise right shift with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("minimum")
.describe("Elementwise minimum of two tensors with broadcasting")
.set_support_level(4);

// Comparisons
#define RELAY_REGISTER_CMP_OP(OpName, SupportLevel)                 \
  TVM_REGISTER_API("relay.op._make." OpName)                        \
  .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {        \
      static const Op& op = Op::Get(OpName);                        \
    return CallNode::make(op, {lhs, rhs}, Attrs(), {});             \
  });                                                               \
  RELAY_REGISTER_OP(OpName)                                         \
    .set_num_inputs(2)                                              \
    .add_argument("lhs", "Tensor", "The left hand side tensor.")    \
    .add_argument("rhs", "Tensor", "The right hand side tensor.")   \
    .set_support_level(SupportLevel)                                \
    .add_type_rel("BroadcastComp", BroadcastCompRel);

RELAY_REGISTER_CMP_OP("equal", 4);
RELAY_REGISTER_CMP_OP("not_equal", 4);
RELAY_REGISTER_CMP_OP("less", 4);
RELAY_REGISTER_CMP_OP("less_equal", 4);
RELAY_REGISTER_CMP_OP("greater", 4);
RELAY_REGISTER_CMP_OP("greater_equal", 4);

}  // namespace relay
}  // namespace tvm
