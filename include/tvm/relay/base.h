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

#include <tvm/ir/span.h>
#include <tvm/node/node.h>
#include <tvm/tir/expr.h>

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

#define RELAY_DEBUG(...)                                               \
  {                                                                    \
    auto fdebug = runtime::Registry::Get("relay.debug");               \
    CHECK(fdebug) << "Could not find Relay Python debugger function."; \
    (*fdebug)("RELAY_DEBUG", __FILE__, __LINE__, __VA_ARGS__);         \
  }

#define RELAY_DEBUG_INTERP(...)                                        \
  {                                                                    \
    auto fdebug = runtime::Registry::Get("relay.debug_interp");        \
    CHECK(fdebug) << "Could not find Relay Python debugger function."; \
    (*fdebug)("RELAY_DEBUG", __FILE__, __LINE__, __VA_ARGS__);         \
  }

/*!
 * \brief Symbolic expression for tensor shape.
 */
using IndexExpr = ::tvm::PrimExpr;

using SourceName = tvm::SourceName;
using Span = tvm::Span;
using SpanNode = tvm::SpanNode;

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
  String name_hint;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name_hint", &name_hint); }

  static constexpr const char* _type_key = "relay.Id";
  TVM_DECLARE_FINAL_OBJECT_INFO(IdNode, Object);
};

class Id : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param name_hint The name of the variable.
   */
  TVM_DLL explicit Id(String name_hint);

  TVM_DEFINE_OBJECT_REF_METHODS(Id, ObjectRef, IdNode);
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BASE_H_
