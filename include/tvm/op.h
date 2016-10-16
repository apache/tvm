/*!
 *  Copyright (c) 2016 by Contributors
 * \file op.h
 * \brief Defines the operators
 */
#ifndef TVM_OP_H_
#define TVM_OP_H_

#include <string>
#include "./expr.h"

namespace tvm {

/*! \brief binary operator */
class BinaryOp {
 public:
  /*! \return the function name to be called in binary op */
  virtual const char* FunctionName() const = 0;
  /*!
   * \brief apply the binary op
   * \param lhs left operand
   * \param rhs right operand
   * \return the result expr
   */
  Expr operator()(Expr lhs, Expr rhs) const;
};


/*! \brief unary operator */
class UnaryOp {
 public:
  /*! \return the function name to be called in unary op */
  virtual const char* FunctionName() const = 0;
  /*!
   * \brief apply the unary op
   * \param src left operand
   * \return the result expr
   */
  Expr operator()(Expr lhs, Expr rhs) const;
};


class AddOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "+";
  }
  static AddOp* Get();
};


class SubOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "-";
  }
  static SubOp* Get();
};


class MulOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "*";
  }
  static MulOp* Get();
};


class DivOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "/";
  }
  static DivOp* Get();
};


class MaxOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "max";
  }
  static MaxOp* Get();
};


class MinOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "min";
  }
  static MinOp* Get();
};

#define DEFINE_OP_OVERLOAD(OpChar, OpName)              \
  inline Expr operator OpChar (Expr lhs, Expr rhs) {    \
    return (*OpName::Get())(lhs, rhs);                  \
  }

#define DEFINE_BINARY_OP_FUNCTION(FuncName, OpName)     \
  inline Expr FuncName(Expr lhs, Expr rhs) {            \
    return (*OpName::Get())(lhs, rhs);                  \
  }

DEFINE_OP_OVERLOAD(+, AddOp);
DEFINE_OP_OVERLOAD(-, SubOp);
DEFINE_OP_OVERLOAD(*, MulOp);
DEFINE_OP_OVERLOAD(/, DivOp);

DEFINE_BINARY_OP_FUNCTION(max, MaxOp);
DEFINE_BINARY_OP_FUNCTION(min, MinOp);

}  // namespace tvm

#endif  // TVM_OP_H_
