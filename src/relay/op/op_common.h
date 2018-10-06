/*!
 *  Copyright (c) 2018 by Contributors
 * \file op_common.h
 * \brief A set of utilities and common functionality
 * for relay ops.
 */
#ifndef TVM_RELAY_OP_OP_COMMON_H_
#define TVM_RELAY_OP_OP_COMMON_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <vector>

namespace tvm {
namespace relay {

template<typename T>
std::vector<T> AsVector(const Array<T> &array) {
    std::vector<T> result;
    result.reserve(array.size());
    for (const T& ele : array) {
        result.push_back(ele);
    }
    return result;
}

/*! Quick helper macro
 * - Expose a positional make function to construct the node.
 * - Register op to the registry.
 *
 * We make the decision to always only expose positional argument.
 * We will do rewrapping in the frontend to support language
 * sugars such as keyword arguments and default value.
 *
 * \param Prefix the prefix of the registry, for example, "relay.op._make.".
 *
 * \param OpName the name of registry.
 */
#define RELAY_REGISTER_UNARY_OP(Prefix, OpName)           \
  TVM_REGISTER_API(Prefix OpName)                         \
    .set_body_typed<Expr(Expr)>([](Expr data) {           \
        static const Op& op = Op::Get(OpName);            \
        return CallNode::make(op, {data}, Attrs(), {});   \
      });                                                 \
  RELAY_REGISTER_OP(OpName)                               \
    .set_num_inputs(1)                                    \
    .add_argument("data", "Tensor", "The input tensor.")

/*! Quick helper macro
 * - Expose a positional make function to construct the node.
 * - Register op to the registry.
 *
 * We make the decision to always only expose positional argument.
 * We will do rewrapping in the frontend to support language
 * sugars such as keyword arguments and default value.
 *
 * \param Prefix the prefix of the registry, for example, "relay.op._make.".
 *
 * \param OpName the name of registry.
 */
#define RELAY_REGISTER_BINARY_OP(Prefix, OpName)                  \
  TVM_REGISTER_API(Prefix OpName)                                 \
    .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {    \
        static const Op& op = Op::Get(OpName);                    \
        return CallNode::make(op, {lhs, rhs}, Attrs(), {});       \
      });                                                         \
  RELAY_REGISTER_OP(OpName)                                       \
    .set_num_inputs(2)                                            \
    .add_argument("lhs", "Tensor", "The left hand side tensor.")  \
    .add_argument("rhs", "Tensor", "The right hand side tensor.") \
    .add_type_rel("Broadcast", BroadcastRel)

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_OP_COMMON_H_
