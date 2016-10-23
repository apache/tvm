/*!
 *  Copyright (c) 2016 by Contributors
 * \file stmt.h
 * \brief Common data structure for codegen
 */
#ifndef TVM_STMT_NODE_H_
#define TVM_STMT_NODE_H_

#include "./base.h"
#include "./domain.h"

namespace tvm {

/*!
 * \brief The internal base class of StmtNode
 *  So far no extra stuffs in here.
 */
struct StmtNode : public Node {
};

/*! \brief Store data into buffer */
struct StoreNode : public StmtNode {
  /*! \brief the variable representing the buffer */
  Var buffer;
  /*! \brief the buffer offset */
  Expr offset;
  /*! \brief The source expression*/
  Expr src;
  /*! \brief constructor */
  StoreNode() {
    node_type_ = kStoreNode;
  }
  const char* type_key() const override {
    return "StoreNode";
  }
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("buffer", &buffer);
    fvisit("offset", &offset);
    fvisit("src", &src);
  }
  void Verify() const override {
    CHECK_EQ(Ptr2DataType(buffer.dtype()), src.dtype());
    CHECK_EQ(offset.dtype(), kInt32);
  }
};

/*! \brief for loop in range */
struct ForRangeNode : public StmtNode {
  /*! \brief loop variable */
  Var loop_var;
  /*! \brief The loop range */
  Range range;
  /*! \brief body of the loop */
  Stmt body;
  /*! \brief constructor */
  ForRangeNode() {
    node_type_ = kForRangeNode;
  }
  const char* type_key() const override {
    return "ForRangeNode";
  }
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("loop_var", &loop_var);
    fvisit("range", &range);
    fvisit("body", &body);
  }
  void Verify() const override {
    CHECK_EQ(loop_var.dtype(), kInt32);
    CHECK_EQ(this->range->begin.dtype(), loop_var.dtype());
    CHECK_EQ(this->range->end.dtype(), loop_var.dtype());
  }
};

/*! \brief conditional expression */
struct IfThenElseNode : public StmtNode {
  /*! \brief The condition */
  Expr cond;
  /*! \brief The statement in then */
  Stmt then_body;
  /*! \brief The statement in else */
  Stmt else_body;
  /*! \brief constructor */
  IfThenElseNode() {
    node_type_ = kIfThenElseNode;
  }
  const char* type_key() const override {
    return "IfThenElseNode";
  }
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("cond", &cond);
    fvisit("then_body", &then_body);
    fvisit("else_body", &else_body);
  }
  void Verify() const override {
    CHECK_EQ(cond.dtype(), kInt32);
  }
};

}  // namespace tvm

#endif  // TVM_STMT_NODE_H_
