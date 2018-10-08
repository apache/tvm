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

RELAY_REGISTER_BINARY_OP("add")
.describe("Elementwise add with with broadcasting")
.set_support_level(1);

// Subtraction
RELAY_REGISTER_BINARY_OP("subtract")
.describe("Elementwise substract with broadcasting")
.set_support_level(1);

// Right shift
RELAY_REGISTER_BINARY_OP("right_shift")
.describe("Elementwise right shift with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("left_shift")
.describe("Elementwise left shift with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("maximum")
.describe("Elementwise maximum of two tensors with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("minimum")
.describe("Elementwise minimum of two tensors with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("divide")
.describe("Elementwise divide with broadcasting")
.set_support_level(1);

RELAY_REGISTER_BINARY_OP("multiply")
.describe("Elementwise multiply with broadcasting")
.set_support_level(1);

RELAY_REGISTER_BINARY_OP("pow")
.describe("Elementwise power with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("mod")
.describe("Elementwise mod with broadcasting")
.set_support_level(1);

// Comparisons
#define RELAY_REGISTER_CMP_OP(OpName)                               \
  TVM_REGISTER_API("relay.op._make." OpName)                        \
  .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {        \
      static const Op& op = Op::Get(OpName);                        \
    return CallNode::make(op, {lhs, rhs}, Attrs(), {});             \
  });                                                               \
  RELAY_REGISTER_OP(OpName)                                         \
    .set_num_inputs(2)                                              \
    .add_argument("lhs", "Tensor", "The left hand side tensor.")    \
    .add_argument("rhs", "Tensor", "The right hand side tensor.")   \
    .add_type_rel("BroadcastComp", BroadcastCompRel)

RELAY_REGISTER_CMP_OP("equal")
.describe("Elementwise equal compare with broadcasting")
.set_support_level(4);
RELAY_REGISTER_CMP_OP("not_equal")
.describe("Elementwise not equal with broadcasting")
.set_support_level(4);
RELAY_REGISTER_CMP_OP("less")
.describe("Elementwise less than with broadcasting")
.set_support_level(4);
RELAY_REGISTER_CMP_OP("less_equal")
.describe("Elementwise less than or equal compare with broadcasting")
.set_support_level(4);
RELAY_REGISTER_CMP_OP("greater")
.describe("Elementwise greater than compare with broadcasting")
.set_support_level(4);
RELAY_REGISTER_CMP_OP("greater_equal")
.describe("Elementwise greater than or equal compare with broadcasting")
.set_support_level(4);

}  // namespace relay
}  // namespace tvm
