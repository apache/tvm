/*!
 *  Copyright (c) 2016 by Contributors
 * \file codegen.h
 * \brief Common data structure for codegen
 */
#ifndef TVM_CODEGEN_H_
#define TVM_CODEGEN_H_

namespace tvm {

// incomplete spec.
struct Assign : public Node {
  Expr src;
  Expr offset;
  Var  ptr;
};

struct Assign : public Node {
  Expr src;
  Expr offset;
  Var  ptr;
};

struct Loop : public Node {
  Expr init;
  Expr cond;
  Stmt body;
};

struct IfThenElse : public Node {
  Expr cond;
  Expr then_;
  Stmt else_;
};


}  // namespace tvm

#endif  // TVM_CODEGEN_H_
