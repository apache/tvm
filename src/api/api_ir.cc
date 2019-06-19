/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to IR build
 * \file api_ir.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/api_registry.h>
#include <tvm/expr_operator.h>

namespace tvm {
namespace ir {

TVM_REGISTER_API("_Var")
.set_body_typed<VarExpr(std::string, Type)>([](std::string s, Type t) {
    return Variable::make(t, s);
  });

TVM_REGISTER_API("make.abs")
.set_body_typed(tvm::abs);

TVM_REGISTER_API("make.floor")
.set_body_typed(tvm::floor);

TVM_REGISTER_API("make.ceil")
.set_body_typed(tvm::ceil);

TVM_REGISTER_API("make.round")
.set_body_typed(tvm::round);

TVM_REGISTER_API("make.trunc")
.set_body_typed(tvm::trunc);

TVM_REGISTER_API("make._cast")
.set_body_typed(tvm::cast);

TVM_REGISTER_API("make._range_by_min_extent")
.set_body_typed(Range::make_by_min_extent);

TVM_REGISTER_API("make.For")
.set_body_typed<Stmt(VarExpr, Expr, Expr, int, int, Stmt)>([](
  VarExpr loop_var, Expr min, Expr extent,
  int for_type, int device_api, Stmt body) {
  return For::make(loop_var,
                   min,
                   extent,
                   static_cast<ForType>(for_type),
                   static_cast<DeviceAPI>(device_api),
                   body);
});

TVM_REGISTER_API("make.Load")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    Type t = args[0];
    if (args.size() == 3) {
      *ret = Load::make(t, args[1], args[2], const_true(t.lanes()));
    } else {
      *ret = Load::make(t, args[1], args[2], args[3]);
    }
  });

TVM_REGISTER_API("make.Store")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    Expr value = args[1];
    if (args.size() == 3) {
      *ret = Store::make(args[0], value, args[2], const_true(value.type().lanes()));
    } else {
      *ret = Store::make(args[0], value, args[2], args[3]);
    }
  });

TVM_REGISTER_API("make.Realize")
.set_body_typed(Realize::make);

TVM_REGISTER_API("make.Call")
.set_body_typed<Expr(Type, std::string, Array<Expr>, int, FunctionRef, int)>([](
  Type type, std::string name,
  Array<Expr> args, int call_type,
  FunctionRef func, int value_index
) {
  return Call::make(type,
                    name,
                    args,
                    static_cast<Call::CallType>(call_type),
                    func,
                    value_index);
});

TVM_REGISTER_API("make.CommReducer")
.set_body_typed(CommReducerNode::make);

// make from two arguments
#define REGISTER_MAKE(Node)                                  \
  TVM_REGISTER_API("make."#Node)                             \
  .set_body_typed(Node::make);                              \

REGISTER_MAKE(Reduce);
REGISTER_MAKE(AttrStmt);

REGISTER_MAKE(IntImm);
REGISTER_MAKE(UIntImm);
REGISTER_MAKE(FloatImm);
REGISTER_MAKE(StringImm);

REGISTER_MAKE(Add);
REGISTER_MAKE(Sub);
REGISTER_MAKE(Mul);
REGISTER_MAKE(Div);
REGISTER_MAKE(Mod);
REGISTER_MAKE(FloorDiv);
REGISTER_MAKE(FloorMod);
REGISTER_MAKE(Min);
REGISTER_MAKE(Max);
REGISTER_MAKE(EQ);
REGISTER_MAKE(NE);
REGISTER_MAKE(LT);
REGISTER_MAKE(LE);
REGISTER_MAKE(GT);
REGISTER_MAKE(GE);
REGISTER_MAKE(And);
REGISTER_MAKE(Or);

REGISTER_MAKE(Not);
REGISTER_MAKE(Select);
REGISTER_MAKE(Ramp);
REGISTER_MAKE(Cast);
REGISTER_MAKE(Broadcast);
REGISTER_MAKE(Shuffle);
REGISTER_MAKE(Let);
REGISTER_MAKE(LetStmt);
REGISTER_MAKE(AssertStmt);
REGISTER_MAKE(ProducerConsumer);
REGISTER_MAKE(Provide);
REGISTER_MAKE(Prefetch);
REGISTER_MAKE(Free);
REGISTER_MAKE(IfThenElse);
REGISTER_MAKE(Evaluate);

// overloaded, needs special handling
TVM_REGISTER_API("make.Block")
  .set_body_typed(static_cast<Stmt (*)(Stmt, Stmt)>(Block::make));

// has default args
TVM_REGISTER_API("make.Allocate")
  .set_body_typed<Stmt(VarExpr, Type, Array<Expr>, Expr, Stmt)>([](
    VarExpr buffer_var, Type type, Array<Expr> extents, Expr condition, Stmt body
  ){
    return Allocate::make(buffer_var, type, extents, condition, body);
  });

// operator overloading, smarter than make
#define REGISTER_MAKE_BINARY_OP(Node, Func)                  \
  TVM_REGISTER_API("make."#Node)                             \
  .set_body_typed<Expr(Expr, Expr)>([](Expr a, Expr b) {     \
      return (Func(a, b));                                   \
    })

#define REGISTER_MAKE_BIT_OP(Node, Func)                                \
  TVM_REGISTER_API("make."#Node)                                        \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                       \
      bool lhs_is_int = args[0].type_code() == kDLInt;                  \
      bool rhs_is_int = args[1].type_code() == kDLInt;                  \
      if (lhs_is_int) {                                                 \
        *ret = (Func(args[0].operator int(), args[1].operator Expr())); \
      } else if (rhs_is_int) {                                          \
        *ret = (Func(args[0].operator Expr(), args[1].operator int())); \
      } else {                                                          \
        *ret = (Func(args[0].operator Expr(), args[1].operator Expr())); \
      }                                                                 \
    })


REGISTER_MAKE_BINARY_OP(_OpAdd, operator+);
REGISTER_MAKE_BINARY_OP(_OpSub, operator-);
REGISTER_MAKE_BINARY_OP(_OpMul, operator*);
REGISTER_MAKE_BINARY_OP(_OpDiv, operator/);
REGISTER_MAKE_BINARY_OP(_OpMod, operator%);
REGISTER_MAKE_BINARY_OP(_OpFloorDiv, floordiv);
REGISTER_MAKE_BINARY_OP(_OpFloorMod, floormod);
REGISTER_MAKE_BINARY_OP(_OpMin, min);
REGISTER_MAKE_BINARY_OP(_OpMax, max);
REGISTER_MAKE_BINARY_OP(_OpEQ, operator==);
REGISTER_MAKE_BINARY_OP(_OpNE, operator!=);
REGISTER_MAKE_BINARY_OP(_OpLT, operator<); // NOLINT(*)
REGISTER_MAKE_BINARY_OP(_OpLE, operator<=); // NOLINT(*)
REGISTER_MAKE_BINARY_OP(_OpGT, operator>);  // NOLINT(*)
REGISTER_MAKE_BINARY_OP(_OpGE, operator>=);
REGISTER_MAKE_BINARY_OP(_OpAnd, operator&&);
REGISTER_MAKE_BINARY_OP(_OpOr, operator||);
REGISTER_MAKE_BIT_OP(bitwise_and, operator&);
REGISTER_MAKE_BIT_OP(bitwise_or, operator|);
REGISTER_MAKE_BIT_OP(bitwise_xor, operator^);
REGISTER_MAKE_BIT_OP(left_shift, operator<<); // NOLINT(*)
REGISTER_MAKE_BIT_OP(right_shift, operator>>);

}  // namespace ir
}  // namespace tvm
