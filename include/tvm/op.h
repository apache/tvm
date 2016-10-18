/*!
 *  Copyright (c) 2016 by Contributors
 * \file op.h
 * \brief Defines the operators
 */
#ifndef TVM_OP_H_
#define TVM_OP_H_

#include <dmlc/registry.h>
#include <string>
#include "./expr.h"
#include "./domain.h"

namespace tvm {

/*! \brief binary operator */
class BinaryOp {
 public:
  // virtual destructor
  virtual ~BinaryOp() {}
  /*! \return the function name to be called in binary op */
  virtual const char* FunctionName() const = 0;
  /*!
   * \brief apply the binary op
   * \param lhs left operand
   * \param rhs right operand
   * \return the result expr
   */
  Expr operator()(Expr lhs, Expr rhs) const;
  /*!
   * \brief make a reduction of src over rdom,
   * \param src Source expression.
   * \param rdom reduction domain.
   * \return the result expr
   */
  Expr Reduce(Expr src, RDomain rdom) const;
  /*!
   * \brief get binary op by name
   * \param name name of operator
   */
  static const BinaryOp* Get(const char* name);
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
  Expr operator()(Expr src) const;
  /*!
   * \brief get unary op by name
   * \param name name of operator
   */
  static const UnaryOp* Get(const char* name);
};


class AddOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "+";
  }
};


class SubOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "-";
  }
};


class MulOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "*";
  }
};


class DivOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "/";
  }
};


class MaxOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "max";
  }
};


class MinOp : public BinaryOp {
 public:
  const char* FunctionName() const override {
    return "min";
  }
};

#define DEFINE_BINARY_OP_OVERLOAD(OpChar)               \
  inline Expr operator OpChar (Expr lhs, Expr rhs) {    \
    static const BinaryOp* op = BinaryOp::Get(#OpChar); \
    return (*op)(lhs, rhs);                             \
  }

#define DEFINE_BINARY_OP_FUNCTION(FuncName)               \
  inline Expr FuncName(Expr lhs, Expr rhs) {              \
    static const BinaryOp* op = BinaryOp::Get(#FuncName); \
    return (*op)(lhs, rhs);                             \
  }

#define DEFINE_REDUCE_FUNCTION(FuncName, OpName)              \
  inline Expr FuncName(Expr src, RDomain rdom) {              \
    static const BinaryOp* op = BinaryOp::Get(#OpName);       \
    return op->Reduce(src, rdom);                             \
  }

DEFINE_BINARY_OP_OVERLOAD(+);
DEFINE_BINARY_OP_OVERLOAD(-);
DEFINE_BINARY_OP_OVERLOAD(*);
DEFINE_BINARY_OP_OVERLOAD(/);

DEFINE_BINARY_OP_FUNCTION(max);
DEFINE_BINARY_OP_FUNCTION(min);

DEFINE_REDUCE_FUNCTION(max, max);
DEFINE_REDUCE_FUNCTION(min, min);
DEFINE_REDUCE_FUNCTION(sum, +);

// overload negation
inline Expr operator-(Expr src) {
  return src * (-1);
}

// template of op registry
template<typename Op>
struct OpReg {
  std::string name;
  std::unique_ptr<Op> op;
  inline OpReg& set(Op* op) {
    this->op.reset(op);
    return *this;
  }
};

using UnaryOpReg = OpReg<UnaryOp>;
using BinaryOpReg = OpReg<BinaryOp>;

#define TVM_REGISTER_BINARY_OP(FunctionName, TypeName)                  \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::BinaryOpReg & __make_ ## _BinOp_ ## TypeName = \
  ::dmlc::Registry<::tvm::BinaryOpReg>::Get()->__REGISTER_OR_GET__(#FunctionName) \
      .set(new TypeName())

#define TVM_REGISTER_UNARY_OP(FunctionName, TypeName)                  \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::BinaryOpReg & __make_ ## _BinOp_ ## TypeName = \
  ::dmlc::Registry<::tvm::UnaryOpReg>::Get()->__REGISTER_OR_GET__(#FunctionName) \
      .set(new TypeName())

}  // namespace tvm

#endif  // TVM_OP_H_
