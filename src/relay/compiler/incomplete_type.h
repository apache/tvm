/*!
 *  Copyright (c) 2018 by Contributors
 * \file incomplete_type.h
 * \brief A way to defined arbitrary function signature with dispatch on types.
 */

#ifndef TVM_RELAY_COMPILER_INCOMPLETE_TYPE_H
#define TVM_RELAY_COMPILER_INCOMPLETE_TYPE_H

#include "tvm/relay/ir.h"

namespace tvm {
namespace relay {

/*!
 * \brief Represents a portion of an incomplete type.
 */
class IncompleteType;

/*! \brief IncompleteType container node */
class IncompleteTypeNode : public TypeNode {
 public:
  TypeParamNode::Kind kind;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("kind", &kind);
  }

  TVM_DLL static IncompleteType make(TypeParamNode::Kind kind);

  static constexpr const char* _type_key = "relay.IncompleteType";
  TVM_DECLARE_NODE_TYPE_INFO(IncompleteTypeNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(IncompleteType, IncompleteTypeNode, Type);

} // namespace relay
} // namespace tvm

#endif  // TVM_RELAY_COMPILER_INCOMPLETE_TYPE_H
