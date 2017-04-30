/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_util.h
 * \brief Helper functions to construct and compose IR nodes.
 */
#ifndef TVM_PASS_IR_UTIL_H_
#define TVM_PASS_IR_UTIL_H_

#include <tvm/ir.h>
#include <vector>

namespace tvm {
namespace ir {

/*!
 * \brief combine the nest stmt, whose body is not defined.
 * \param nest A list of For and LetStmt, whose body is not defined.
 * \param body body
 * \return The combined Stmt
 */
inline Stmt MergeNest(std::vector<Stmt> nest, Stmt body) {
  // use reverse iteration
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    Stmt s = *ri;
    if (s.as<For>()) {
      auto n = std::make_shared<For>(*s.as<For>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<LetStmt>()) {
      auto n = std::make_shared<LetStmt>(*s.as<LetStmt>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<AttrStmt>()) {
      auto n = std::make_shared<AttrStmt>(*s.as<AttrStmt>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<IfThenElse>()) {
      auto n = std::make_shared<IfThenElse>(*s.as<IfThenElse>());
      CHECK(is_no_op(n->then_case));
      CHECK(!n->else_case.defined());
      n->then_case = body;
      body = Stmt(n);
    } else if (s.as<AssertStmt>()) {
      body = Block::make(s, body);
    } else if (s.as<Allocate>()) {
      auto n = std::make_shared<Allocate>(*s.as<Allocate>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else {
      LOG(FATAL) << "not supported nest type";
    }
  }
  return body;
}

/*!
 * \brief combine the nest stmt, whose body is not defined.
 * \param nest A list of For and LetStmt, whose body is not defined.
 * \param body body
 * \return The combined Stmt
 */
inline Stmt MergeNest(std::vector<std::vector<Stmt> > nest, Stmt body) {
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    body = MergeNest(*ri, body);
  }
  return body;
}


/*!
 * \brief combine sequence of operations.
 * \param seq The sequence.
 * \return The combined Stmt
 */
inline Stmt MergeSeq(const std::vector<Stmt>& seq) {
  if (seq.size() == 0) return Evaluate::make(0);
  Stmt body = seq[0];
  for (size_t i = 1; i < seq.size(); ++i) {
    body = Block::make(body, seq[i]);
  }
  return body;
}

/*!
 * \brief Get construct from struct
 * \param dtype The data type.
 * \param handle the struct handle.
 * \param index the offset index.
 * \param kind The data kind.
 * \return the get expression.
 */
inline Expr TVMStructGet(
    Type dtype, Var handle, int index,
    intrinsic::TVMStructFieldKind kind) {
  Array<Expr> args ={
    handle,
    make_const(Int(32), index),
    make_const(Int(32), kind)};
  return Call::make(dtype, intrinsic::tvm_struct_get, args, Call::PureIntrinsic);
}

/*!
 * \brief Address of handle + offset
 * \param handle the array handle.
 * \param dtype The data type.
 * \param offset the offset index.
 */
inline Expr AddressOffset(Var handle, Type dtype, int offset) {
  return Call::make(
      Handle(), Call::address_of,
      {Load::make(dtype, handle, make_const(Int(32), offset))}, Call::PureIntrinsic);
}

/*!
 * \brief Set value into struct.
 * \param handle the struct handle.
 * \param index the offset index.
 * \param kind The data kind.
 * \param value The value to be set.
 * \return the set stmt.
 */
inline Stmt TVMStructSet(
    Var handle, int index,
    intrinsic::TVMStructFieldKind kind, Expr value) {
  Array<Expr> args ={
    handle,
    make_const(Int(32), index),
    make_const(Int(32), kind),
    value};
  return Evaluate::make(
      Call::make(Int(32), intrinsic::tvm_struct_set, args, Call::Intrinsic));
}

/*!
 * \brief Get the type that is passed around TVM PackedFunc API.
 * \param t The original type.
 * \return The corresponding API type.
 */
inline Type APIType(Type t) {
  if (t.is_handle()) return t;
  CHECK_EQ(t.lanes(), 1)
      << "Cannot pass vector type through packed API.";
  if (t.is_uint() || t.is_int()) return Int(64);
  CHECK(t.is_float());
  return Float(64);
}

}  // namespace ir
}  // namespace tvm
#endif  // TVM_PASS_IR_UTIL_H_
