/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to IR build
 * \file api_ir.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <ir/IROperator.h>
#include <tvm/api_registry.h>

namespace tvm {
namespace ir {

TVM_REGISTER_API("_Var")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = Variable::make(args[1], args[0]);
  });

TVM_REGISTER_API("make.For")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = For::make(args[0],
                     args[1],
                     args[2],
                     static_cast<ForType>(args[3].operator int()),
                     static_cast<Halide::DeviceAPI>(args[4].operator int()),
                     args[5]);
  });

TVM_REGISTER_API("make.Realize")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = Realize::make(args[0],
                         args[1],
                         args[2],
                         args[3],
                         args[4],
                         args[5]);
  });


TVM_REGISTER_API("make.Call")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = Call::make(args[0],
                      args[1],
                      args[2],
                      static_cast<Call::CallType>(args[3].operator int()),
                      args[4],
                      args[5]);
  });

TVM_REGISTER_API("make.Allocate")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = Allocate::make(args[0],
                          args[1],
                          args[2],
                          args[3],
                          args[4]);
  });

// make from two arguments
#define REGISTER_MAKE1(Node)                                 \
  TVM_REGISTER_API("make."#Node)                             \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {            \
      *ret = Node::make(args[0]);                            \
    })                                                       \

#define REGISTER_MAKE2(Node)                                 \
  TVM_REGISTER_API("make."#Node)                             \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {            \
      *ret = Node::make(args[0], args[1]);                   \
    })                                                       \

#define REGISTER_MAKE3(Node)                                 \
  TVM_REGISTER_API("make."#Node)                             \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {            \
      *ret = Node::make(args[0], args[1], args[2]);          \
    })                                                       \

#define REGISTER_MAKE4(Node)                                            \
  TVM_REGISTER_API("make."#Node)                                        \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                       \
      *ret = Node::make(args[0], args[1], args[2], args[3]);            \
    })                                                                  \

#define REGISTER_MAKE5(Node)                                            \
  TVM_REGISTER_API("make."#Node)                                        \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                       \
      *ret = Node::make(args[0], args[1],                               \
                        args[2], args[3], args[4]);                     \
    })                                                                  \

#define REGISTER_MAKE_BINARY_OP(Node)                        \
  TVM_REGISTER_API("make."#Node)                             \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {            \
      Expr a = args[0], b = args[1];                         \
      match_types(a, b);                                     \
      *ret = Node::make(a, b);                               \
    })

REGISTER_MAKE2(Functor);
REGISTER_MAKE4(AttrStmt);

TVM_REGISTER_API("make.Reducer")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = Reduce::make(
      static_cast<std::string>(args[0].operator std::string()),
      args[1], args[2], args[3]);
  });

TVM_REGISTER_API("make.CommReducer")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = Reduce::make(
      static_cast<Functor>(args[0].operator Functor()),
      args[1], args[2], args[3], args[4]);
  });


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
REGISTER_MAKE3(Load);
REGISTER_MAKE3(Store);
REGISTER_MAKE4(Provide);
REGISTER_MAKE1(Free);
REGISTER_MAKE2(Block);
REGISTER_MAKE3(IfThenElse);
REGISTER_MAKE1(Evaluate);

}  // namespace ir
}  // namespace tvm
