/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/base.h
 * \brief Base data structure for relay.
 */
#ifndef TVM_RELAY_BASE_H_
#define TVM_RELAY_BASE_H_

#include <tvm/api_registry.h>
#include <tvm/ir.h>
#include <tvm/node.h>
#include <string>

namespace tvm {
/*!
 * \brief Relay: high level functional IR
 */
namespace relay {
/*!
 * \brief we always used NodeRef for referencing nodes.
 *
 *  By default, NodePtr is a std::shared_ptr of node
 */
using NodeRef = tvm::NodeRef;

/*!
 * \brief Content data type.
 */
using DataType = ::tvm::Type;

/*!
 * \brief Symbolic expression for tensor shape.
 */
using ShapeExpr = ::tvm::Expr;

/*!
 * \brief Hash function for nodes.
 * e.g. std::unordered_map<Expr, Value, NodeHash, NodeEqual>
 */
using NodeHash = ::tvm::NodeHash;
/*!
 * \brief Equality check function for nodes.
 */
using NodeEqual = ::tvm::NodeEqual;

/*!
 * \brief Macro to make it easy to define node ref type given node
 * \param TypeName The name of the reference type.
 * \param NodeName The internal contrainer name.
 * \param NodeRefBase The base type.
 */
#define RELAY_DEFINE_NODE_REF(TypeName, NodeName, NodeRefBase)          \
  class TypeName : public NodeRefBase {                                 \
   public:                                                              \
    TypeName() {}                                                        \
    explicit TypeName(std::shared_ptr<::tvm::Node> n) : NodeRefBase(n) {} \
    const NodeName* operator->() const {                                 \
      return static_cast<const NodeName*>(node_.get());                  \
    }                                                                    \
    using ContainerType = NodeName;                                      \
  };


/*!
 * \brief The source name in the Span
 * \sa SourceNameNode, Span
 */
class SourceName;
/*!
 * \brief The source name in the Span
 */
class SourceNameNode : public Node {
 public:
  /*! \brief The source name */
  std::string name;
  // override attr visitor
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
  }

  TVM_DLL static SourceName make(std::string name);

  static constexpr const char* _type_key = "relay.SourceName";
  TVM_DECLARE_NODE_TYPE_INFO(SourceNameNode, Node);
};

RELAY_DEFINE_NODE_REF(SourceName, SourceNameNode, NodeRef);

/*!
 * \brief Span information for debugging purposes
 */
class Span;
/*!
 * \brief Stores locations in frontend source that generated a node.
 *
 */
class SpanNode : public Node {
 public:
  /*! \brief The source name */
  SourceName source;
  /*! \brief Line number */
  int lineno;
  /*! \brief column offset */
  int col_offset;
  // override attr visitor
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("source", &source);
    v->Visit("lineno", &lineno);
    v->Visit("col_offset", &col_offset);
  }

  TVM_DLL static Span make(SourceName source, int lineno, int col_offset);

  static constexpr const char* _type_key = "relay.Span";
  TVM_DECLARE_NODE_TYPE_INFO(SpanNode, Node);
};

RELAY_DEFINE_NODE_REF(Span, SpanNode, NodeRef);

/*!
 * \brief This is the base node container of all relay structures.
 */
class RelayNode : public Node {
 public:
  /*! \brief The debug information, can be null, check with span.defined() */
  mutable Span span;

  static constexpr const char* _type_key = "relay.Node";
  TVM_DECLARE_BASE_NODE_INFO(RelayNode, Node);
};

/*!
 * \brief Get a reference type from a Node ptr type
 *
 *  It is always important to get a reference type
 *  if we want to return a value as reference or keep
 *  the node alive beyond the scope of the function.
 *
 * \param ptr The node pointer
 * \tparam RefType The reference type
 * \tparam NodeType The node type
 * \return The corresponding RefType
 */
template <typename RefType, typename NodeType>
RefType GetRef(const NodeType* ptr) {
  static_assert(std::is_same<typename RefType::ContainerType, NodeType>::value,
                "Can only cast to the ref of same container type");
  return RefType(const_cast<NodeType*>(ptr)->shared_from_this());
}

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BASE_H_
