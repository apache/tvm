/*!
 *  Copyright (c) 2017 by Contributors
 * \file ir_operator.cc
 */
#include <tvm/base.h>
#include <tvm/ir.h>
#include <tvm/ir_operator.h>

namespace tvm {

Expr sum(Expr source, Array<IterVar> rdom) {
  Var x("x", source.type()), y("y", source.type());
  Expr result = ir::Add::make(x, y);
  Expr identity_element = make_zero(source.type());
  ir::CommReducer combiner =
    ir::CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return ir::Reduce::make(combiner, {source}, rdom, make_const(Bool(1), true), 0);
}

Expr max(Expr source, Array<IterVar> rdom) {
  Var x("x", source.type()), y("y", source.type());
  Expr result = ir::Max::make(x, y);
  Expr identity_element = source.type().min();
  ir::CommReducer combiner =
    ir::CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return ir::Reduce::make(combiner, {source}, rdom, make_const(Bool(1), true), 0);
}

Expr min(Expr source, Array<IterVar> rdom) {
  Var x("x", source.type()), y("y", source.type());
  Expr result = ir::Min::make(x, y);
  Expr identity_element = source.type().max();
  ir::CommReducer combiner =
    ir::CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return ir::Reduce::make(combiner, {source}, rdom, make_const(Bool(1), true), 0);
}

Expr prod(Expr source, Array<IterVar> rdom) {
  Var x("x", source.type()), y("y", source.type());
  Expr result = ir::Mul::make(x, y);
  Expr identity_element = make_one(source.type());
  ir::CommReducer combiner =
    ir::CommReducerNode::make({x}, {y}, {result}, {identity_element});
  return ir::Reduce::make(combiner, {source}, rdom, make_const(Bool(1), true), 0);
}

}  // namespace tvm
