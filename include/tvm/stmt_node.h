/*!
 *  Copyright (c) 2016 by Contributors
 * \file stmt.h
 * \brief Common data structure for codegen
 */
#ifndef TVM_STMT_NODE_H_
#define TVM_STMT_NODE_H_

namespace tvm {

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
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("buffer", &buffer);
    fvisit("offset", &offset);
    fvisit("src", &src);
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
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("loop_var", &loop_var);
    fvisit("range", &range);
    fvisit("body", &body);
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
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("cond", &cond);
    fvisit("then_body", &then_body);
    fvisit("else_body", &else_body);
  }
};

}  // namespace tvm

#endif  // TVM_CODEGEN_H_
