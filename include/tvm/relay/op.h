/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/op.h
 * \brief Relay's representation of operators.
 */
#ifndef TVM_RELAY_OP_H_
#define TVM_RELAY_OP_H_

#include "./expr.h"

namespace tvm {
namespace relay {


/*!
 * \brief A primitive Relay operator defined externally to Relay.
 *
 * \note Currently these are expected to be backed by a TVM's operator,
 * such as the ones defined in TOPI.
 *
 *  For developers who are familar with the computational graph this
 *  directly maps to the concept of operators in NNVM.
 */
class Operator;
/*! \brief Container for Operator */
class OperatorNode : public ExprNode {
 public:
  /*! \brief A type which specifies the relationship between the inputs and outputs
   *  of the operator.
   */
  Type op_type;

  void VisitAttrs(tvm::AttrVisitor* v) final {
      v->Visit("op_type", &op_type);
  }

  TVM_DLL static Operator make(Type op_type);

  static constexpr const char* _type_key = "relay.Operator";
  TVM_DECLARE_NODE_TYPE_INFO(OperatorNode, OperatorNode);
};

RELAY_DEFINE_NODE_REF(Operator, OperatorNode, Expr);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_H_
