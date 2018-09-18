/*!
 *  Copyright (c) 2018 by Contributors
 * \file builder.h
 * \brief A helper file that make building expr easier.
 */
#ifndef TVM_RELAY_BUILDER_H_
#define TVM_RELAY_BUILDER_H_

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

inline Expr GetField(const Expr& t, size_t i) {
  return TupleGetItemNode::make(t, i);
}

inline Expr Mult(const Expr& l, const Expr& r) {
  static const Op& op = Op::Get("multiply");
  return CallNode::make(op, {l, r});
}

inline Expr Div(const Expr& l, const Expr& r) {
  static const Op& op = Op::Get("divide");
  return CallNode::make(op, {l, r});
}

inline Expr Plus(const Expr& l, const Expr& r) {
  static const Op& op = Op::Get("add");
  return CallNode::make(op, {l, r});
}

inline Expr Pair(const Expr& l, const Expr& r) {
  return TupleNode::make({l, r});
}

inline Expr ZeroLike(const Expr& e) {
  static const Op& op = Op::Get("zeros_like");
  return CallNode::make(op, {e});
}

inline Expr OneLike(const Expr& e) {
  static const Op& op = Op::Get("ones_like");
  return CallNode::make(op, {e});
}

inline Expr Neg(const Expr& e) {
  static const Op& op = Op::Get("neg");
  return CallNode::make(op, {e});
}

inline Expr Sqrt(const Expr& e) {
  static const Op& op = Op::Get("sqrt");
  return CallNode::make(op, {e});
}

inline Expr Exp(const Expr& e) {
  static const Op& op = Op::Get("exp");
  return CallNode::make(op, {e});
}

inline Expr Log(const Expr& e) {
  static const Op& op = Op::Get("log");
  return CallNode::make(op, {e});
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BUILDER_H_
