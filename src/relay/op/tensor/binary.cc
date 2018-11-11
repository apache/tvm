/*!
 *  Copyright (c) 2018 by Contributors
 * \file binary.cc
 * \brief binary broadcast operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include "../type_relations.h"
#include "../op_common.h"

namespace tvm {
namespace relay {

// Addition
RELAY_REGISTER_BINARY_OP("relay.op._make.", "add")
.describe("Elementwise add with with broadcasting")
.set_support_level(1);

// Subtraction
RELAY_REGISTER_BINARY_OP("relay.op._make.", "subtract")
.describe("Elementwise substract with broadcasting")
.set_support_level(1);

// Right shift
RELAY_REGISTER_BINARY_OP("relay.op._make.", "right_shift")
.describe("Elementwise right shift with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("relay.op._make.", "left_shift")
.describe("Elementwise left shift with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("relay.op._make.", "maximum")
.describe("Elementwise maximum of two tensors with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("relay.op._make.", "minimum")
.describe("Elementwise minimum of two tensors with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("relay.op._make.", "divide")
.describe("Elementwise divide with broadcasting")
.set_support_level(1);

RELAY_REGISTER_BINARY_OP("relay.op._make.", "multiply")
.describe("Elementwise multiply with broadcasting")
.set_support_level(1);

RELAY_REGISTER_BINARY_OP("relay.op._make.", "power")
.describe("Elementwise power with broadcasting")
.set_support_level(4);

RELAY_REGISTER_BINARY_OP("relay.op._make.", "mod")
.describe("Elementwise mod with broadcasting")
.set_support_level(1);

// Comparisons
#define RELAY_REGISTER_CMP_OP(OpName)                               \
  TVM_REGISTER_API("relay.op._make." OpName)                        \
  .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {        \
    static const Op& op = Op::Get(OpName);                          \
    return CallNode::make(op, {lhs, rhs}, Attrs(), {});             \
  });                                                               \
  RELAY_REGISTER_OP(OpName)                                         \
    .set_num_inputs(2)                                              \
    .add_argument("lhs", "Tensor", "The left hand side tensor.")    \
    .add_argument("rhs", "Tensor", "The right hand side tensor.")   \
    .add_type_rel("BroadcastComp", BroadcastCompRel)                \
    .set_attr<TOpPattern>("TOpPattern", kBroadcast)

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
