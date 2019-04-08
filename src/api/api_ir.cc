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
.set_body_simple(tvm::abs);

TVM_REGISTER_API("make.floor")
.set_body_simple(tvm::floor);

TVM_REGISTER_API("make.ceil")
.set_body_simple(tvm::ceil);

TVM_REGISTER_API("make.round")
.set_body_simple(tvm::round);

TVM_REGISTER_API("make.trunc")
.set_body_simple(tvm::trunc);

TVM_REGISTER_API("make._cast")
.set_body_simple(tvm::cast);

TVM_REGISTER_API("make._range_by_min_extent")
.set_body_simple(Range::make_by_min_extent);

TVM_REGISTER_API("make.For")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = For::make(args[0],
                     args[1],
                     args[2],
                     static_cast<ForType>(args[3].operator int()),
                     static_cast<HalideIR::DeviceAPI>(args[4].operator int()),
                     args[5]);
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
.set_body_simple(Realize::make);

TVM_REGISTER_API("make.Call")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = Call::make(args[0],
                      args[1],
                      args[2],
                      static_cast<Call::CallType>(args[3].operator int()),
                      args[4],
                      args[5]);
  });

TVM_REGISTER_API("make.CommReducer")
.set_body_simple(CommReducerNode::make);

// make from two arguments
#define REGISTER_MAKE(Node)                                  \
  TVM_REGISTER_API("make."#Node)                             \
  .set_body_simple(Node::make);                              \

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
  .set_body_simple(static_cast<Stmt (*)(Stmt, Stmt)>(Block::make));

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
