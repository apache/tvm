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

#define RELAY_DEBUG_INTERP(...) \
{ auto fdebug = runtime::Registry::Get("relay.debug_interp"); \
  CHECK(fdebug) << "Could not find Relay Python debugger function."; \
  (*fdebug)("RELAY_DEBUG", __FILE__, __LINE__, __VA_ARGS__); \
}

/*!
 * \brief Symbolic expression for tensor shape.
 */
using IndexExpr = ::tvm::Expr;

/*!
 * \brief The source name in the Span
 * \sa SourceNameNode, Span
 */
class SourceName;
/*!
 * \brief The name of a source fragment.
 */
class SourceNameNode : public Object {
 public:
  /*! \brief The source name. */
  std::string name;
  // override attr visitor
  void VisitAttrs(AttrVisitor* v) { v->Visit("name", &name); }

  static constexpr const char* _type_key = "relay.SourceName";
  TVM_DECLARE_FINAL_OBJECT_INFO(SourceNameNode, Object);
};

/*!
 * \brief The source name of a file span.
 * \sa SourceNameNode, Span
 */
class SourceName : public ObjectRef {
 public:
  /*! \brief default constructor  */
  SourceName() {}

  /*! \brief constructor from node pointer */
  explicit SourceName(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const SourceNameNode* operator->() const {
    return static_cast<const SourceNameNode*>(get());
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
class SpanNode : public Object {
 public:
  /*! \brief The source name */
  SourceName source;
  /*! \brief Line number */
  int lineno;
  /*! \brief column offset */
  int col_offset;
  // override attr visitor
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("source", &source);
    v->Visit("lineno", &lineno);
    v->Visit("col_offset", &col_offset);
  }

  TVM_DLL static Span make(SourceName source, int lineno, int col_offset);

  static constexpr const char* _type_key = "relay.Span";
  TVM_DECLARE_FINAL_OBJECT_INFO(SpanNode, Object);
};

class Span : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Span, ObjectRef, SpanNode);
};

/*!
 * \brief This is the base node container of all relay structures.
 */
class RelayNode : public Object {
 public:
  /*! \brief The location of the program in a SourceFragment can be null,
   * check with span.defined() */
  mutable Span span;

  static constexpr const char* _type_key = "relay.Node";
  TVM_DECLARE_BASE_OBJECT_INFO(RelayNode, Object);
};

/*!
 * \brief The unique identifier of variables.
 *
 * Id is like name to the variables,
 * except that id is unique for each Var.
 *
 * \note Do not create Id directly, they are created in Var.
 */
class IdNode : public Object {
 public:
  /*!
   * \brief The name of the variable,
   *  this only acts as a hint to the user,
   *  and is not used for equality.
   */
  std::string name_hint;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
  }

  static constexpr const char* _type_key = "relay.Id";
  TVM_DECLARE_FINAL_OBJECT_INFO(IdNode, Object);
};

class Id : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Id, ObjectRef, IdNode);
};


struct Module;

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BASE_H_
