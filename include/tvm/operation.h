/*!
 *  Copyright (c) 2016 by Contributors
 * \file operation.h
 * \brief Operation node can generate one or multiple Tensors
 */
#ifndef TVM_OPERATION_H_
#define TVM_OPERATION_H_

#include <string>
#include "./expr.h"
#include "./domain.h"

namespace tvm {

// internal node container for Operation
class OperationNode;

/*! \brief Split over input domain */
class Operation : public NodeRef {
 public:
  /*! \brief default constructor  */
  Operation() {}
  explicit Operation(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const OperationNode* operator->() const;
};

/*!
 * \brief base class of operation node.
 */
class OperationNode : public Node {
 public:
  /*! \brief The domain of iteration of this op. */
  Domain domain;
  /*! \brief optional name of the operation */
  std::string name;
};

/*!
 * \brief A Compute op that compute a tensor on certain domain.
 */
class ComputeOpNode : public OperationNode {
 public:
  /*! \brief iter-Var over the dimensions */
  Array<Var> dim_var;
  /*! \brief the compute expression */
  Expr body;
  /*! \brief constructor */
  ComputeOpNode() {}

  const char* type_key() const final {
    return "ComputeOp";
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("domain", &domain);
    v->Visit("name", &name);
    v->Visit("dim_var", &dim_var);
    v->Visit("body", &body);
  }
  static Operation make(Domain domain,
                        std::string name,
                        Array<Var> dim_var,
                        Expr body);
};

// Implementations of inline functions
inline const OperationNode* Operation::operator->() const {
  return static_cast<const OperationNode*>(node_.get());
}

}  // namespace tvm

#endif  // TVM_OPERATION_H_
