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

class BinaryOp {
 public:
  virtual std::string Format(const std::string& lhs, const std::string& rhs);
};

class UnaryOp {
 public:
};

class AddOp : public BinaryOp {
 public:
  static AddOp* Get();
};

class SubOp : public BinaryOp {
 public:
  static SubOp* Get();
};

class MulOp : public BinaryOp {
 public:
  static SubOp* Get();
};

class DivOp : public BinaryOp {
 public:
  static DivOp* Get();
};

}  // namespace tvm

#endif  // TVM_OP_H_
