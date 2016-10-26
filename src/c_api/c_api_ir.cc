/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to IR build
 * \file c_api_ir.cc
 */
#include <tvm/expr.h>
#include <ir/IROperator.h>
#include "./c_api_registry.h"

namespace tvm {

using namespace Halide::Internal;

using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

// make from two arguments
#define REGISTER_MAKE1(Node)                                 \
  TVM_REGISTER_API(_make_## Node)                            \
  .set_body([](const ArgStack& args,  RetValue *ret) {       \
      *ret = Node::make(args.at(0));                         \
    })                                                       \

#define REGISTER_MAKE2(Node)                                 \
  TVM_REGISTER_API(_make_## Node)                            \
  .set_body([](const ArgStack& args,  RetValue *ret) {       \
      *ret = Node::make(args.at(0), args.at(1));             \
    })                                                       \

#define REGISTER_MAKE3(Node)                                 \
  TVM_REGISTER_API(_make_## Node)                            \
  .set_body([](const ArgStack& args,  RetValue *ret) {       \
      *ret = Node::make(args.at(0), args.at(1), args.at(2)); \
    })                                                       \

#define REGISTER_MAKE_BINARY_OP(Node)                            \
  TVM_REGISTER_API(_make_## Node)                            \
  .set_body([](const ArgStack& args,  RetValue *ret) {       \
      Expr a = args.at(0), b = args.at(1);                   \
      match_types(a, b);                                     \
      *ret = Node::make(a, b);                               \
    })                                                       \
  .add_argument("lhs", "Expr", "left operand")               \
  .add_argument("rhs", "Expr", "right operand")

REGISTER_MAKE2(IntImm);
REGISTER_MAKE2(UIntImm);
REGISTER_MAKE2(FloatImm);
REGISTER_MAKE1(StringImm);
REGISTER_MAKE_BINARY_OP(Add);
REGISTER_MAKE_BINARY_OP(Sub);
REGISTER_MAKE_BINARY_OP(Mul);
REGISTER_MAKE_BINARY_OP(Div);
REGISTER_MAKE_BINARY_OP(Mod);
REGISTER_MAKE_BINARY_OP(Min);
REGISTER_MAKE_BINARY_OP(Max);
REGISTER_MAKE_BINARY_OP(EQ);
REGISTER_MAKE_BINARY_OP(NE);
REGISTER_MAKE_BINARY_OP(LT);
REGISTER_MAKE_BINARY_OP(LE);
REGISTER_MAKE_BINARY_OP(GT);
REGISTER_MAKE_BINARY_OP(GE);
REGISTER_MAKE_BINARY_OP(And);
REGISTER_MAKE_BINARY_OP(Or);
REGISTER_MAKE1(Not);
REGISTER_MAKE3(Select);
REGISTER_MAKE3(Ramp);
REGISTER_MAKE2(Broadcast);
REGISTER_MAKE3(Let);
REGISTER_MAKE3(LetStmt);
REGISTER_MAKE2(AssertStmt);
REGISTER_MAKE3(ProducerConsumer);
// TODO(tqchen) For;
REGISTER_MAKE3(Store);
// TODO(tqchen) Provide;
// TODO(tqchen) Allocate;
REGISTER_MAKE1(Free);
// TODO(tqchen) Realize;
REGISTER_MAKE2(Block);
REGISTER_MAKE3(IfThenElse);
REGISTER_MAKE1(Evaluate);

}  // namespace tvm
