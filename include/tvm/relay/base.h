/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relay/base.h
 * \brief Base classes for the Relay IR.
 */
#ifndef TVM_RELAY_BASE_H_
#define TVM_RELAY_BASE_H_

#include <tvm/api_registry.h>
#include <tvm/ir.h>
#include <tvm/node/node.h>
#include <string>
#include <vector>

namespace tvm {
/*!
 * \brief Relay: a high level functional IR for TVM.
 *
 * This namespace contains the abstract syntax tree, and other
 * essential data structures for the Relay IR.
 *
 * You can find more about Relay by reading the language reference.
 */
namespace relay {

#define RELAY_DEBUG(...) \
{ auto fdebug = runtime::Registry::Get("relay.debug"); \
  CHECK(fdebug) << "Could not find Relay Python debugger function."; \
  (*fdebug)("RELAY_DEBUG", __FILE__, __LINE__, __VA_ARGS__); \
}

/*!
 * \brief We always used NodeRef for referencing nodes.
 *
 *  By default, NodeRef is a std::shared_ptr of node
 */
using NodeRef = tvm::NodeRef;

/*!
 * \brief Content data type.
 */
using DataType = ::tvm::Type;

/*!
 * \brief Symbolic expression for tensor shape.
 */
using IndexExpr = ::tvm::Expr;

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
 * \param NodeName The internal container name.
 * \param NodeRefBase The base type.
 */
#define RELAY_DEFINE_NODE_REF(TypeName, NodeName, NodeRefBase)          \
  class TypeName : public NodeRefBase {                                 \
   public:                                                              \
    TypeName() {}                                                        \
    explicit TypeName(::tvm::NodePtr<::tvm::Node> n) : NodeRefBase(n) {} \
    const NodeName* operator->() const {                                \
      return static_cast<const NodeName*>(node_.get());                 \
    }                                                                   \
    operator bool() { return this->defined(); }                         \
    using ContainerType = NodeName;                                     \
  };

/*!
 * \brief The source name in the Span
 * \sa SourceNameNode, Span
 */
class SourceName;
/*!
 * \brief The name of a source fragment.
 */
class SourceNameNode : public Node {
 public:
  /*! \brief The source name. */
  std::string name;
  // override attr visitor
  void VisitAttrs(AttrVisitor* v) final { v->Visit("name", &name); }

  static constexpr const char* _type_key = "relay.SourceName";
  TVM_DECLARE_NODE_TYPE_INFO(SourceNameNode, Node);
};

/*!
 * \brief The source name of a file span.
 * \sa SourceNameNode, Span
 */
class SourceName : public NodeRef {
 public:
  /*! \brief default constructor  */
  SourceName() {}

  /*! \brief constructor from node pointer */
  explicit SourceName(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const SourceNameNode* operator->() const {
    return static_cast<SourceNameNode*>(this->node_.get());
  }

  /*!
   * \brief Get an SourceName for a given operator name.
   *  Will raise an error if the source name has not been registered.
   * \param name Name of the operator.
   * \return SourceName valid throughout program lifetime.
   */
  TVM_DLL static SourceName Get(const std::string& name);

  /*! \brief specify container node */
  using ContainerType = SourceNameNode;
};

/*!
 * \brief Span information for debugging purposes
 */
class Span;
/*!
 * \brief Stores locations in frontend source that generated a node.
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
  /*! \brief The location of the program in a SourceFragment can be null,
   * check with span.defined() */
  mutable Span span;

  static constexpr const char* _type_key = "relay.Node";
  TVM_DECLARE_BASE_NODE_INFO(RelayNode, Node);
};

/*!
 * \brief The unique identifier of variables.
 *
 * Id is like name to the variables,
 * except that id is unique for each Var.
 *
 * \note Do not create Id directly, they are created in Var.
 */
class IdNode : public Node {
 public:
  /*!
   * \brief The name of the variable,
   *  this only acts as a hint to the user,
   *  and is not used for equality.
   */
  std::string name_hint;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name_hint", &name_hint);
  }

  static constexpr const char* _type_key = "relay.Id";
  TVM_DECLARE_NODE_TYPE_INFO(IdNode, Node);
};

RELAY_DEFINE_NODE_REF(Id, IdNode, NodeRef);


struct Module;

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BASE_H_
