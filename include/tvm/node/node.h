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
 * \file tvm/node/node.h
 * \brief Definitions and helper macros for IR/AST nodes.
 *
 *  The node folder contains base utilities for IR/AST nodes,
 *  invariant of which specific language dialect.
 *
 *  We implement AST/IR nodes as sub-classes of runtime::Object.
 *  The base class Node is just an alias of runtime::Object.
 *
 *  Besides the runtime type checking provided by Object,
 *  node folder contains additional functionalities such as
 *  reflection and serialization, which are important features
 *  for building a compiler infra.
 */
#ifndef TVM_NODE_NODE_H_
#define TVM_NODE_NODE_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/memory.h>
#include <tvm/node/reflection.h>

#include <string>
#include <vector>
#include <utility>
#include <type_traits>

namespace tvm {

using runtime::TypeIndex;
using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;
using runtime::GetRef;
using runtime::Downcast;
using runtime::ObjectHash;
using runtime::ObjectEqual;
using runtime::make_object;

using NodeHash = ObjectHash;
using NodeEqual = ObjectEqual;
using Node = Object;

/*!
 * \brief Base class of all references to AST/IR nodes.
 */
class NodeRef : public ObjectRef {
 public:
  NodeRef() {}
  explicit NodeRef(ObjectPtr<Object> n) : ObjectRef(n) {}
};

/*!
 * \brief Allocate a node object.
 * \param args arguments to the constructor.
 * \tparam T the node type.
 * \return The NodePtr to the allocated object.
 * \note This function is an alias of make_object.
 */
template<typename T, typename... Args>
inline NodePtr<T> make_node(Args&&... args) {
  return runtime::make_object<T>(std::forward<Args>(args)...);
}

/*!
 * \brief helper macro to declare type information in a base node.
 */
#define TVM_DECLARE_BASE_NODE_INFO(TypeName, Parent)  \
  TVM_DECLARE_BASE_OBJECT_INFO(TypeName, Parent)

/*!
 * \brief helper macro to declare type information in a terminal node
 */
#define TVM_DECLARE_NODE_TYPE_INFO(TypeName, Parent)  \
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeName, Parent);


/*!
 * \brief Macro to define common node ref methods.
 * \param TypeName The name of the NodeRef.
 * \param BaseTypeName The Base type.
 * \param NodeName The node container type.
 */
#define TVM_DEFINE_NODE_REF_METHODS(TypeName, BaseTypeName, NodeName)   \
  TypeName() {}                                                         \
  explicit TypeName(::tvm::ObjectPtr<::tvm::Object> n)                  \
      : BaseTypeName(n) {}                                              \
  const NodeName* operator->() const {                                  \
    return static_cast<const NodeName*>(data_.get());                   \
  }                                                                     \
  operator bool() const { return this->defined(); }                     \
  using ContainerType = NodeName;

/*!
 * \brief Macro to define CopyOnWrite function in a NodeRef.
 * \param NodeName The Type of the Node.
 *
 *  CopyOnWrite will generate a unique copy of the internal node.
 *  The node will be copied if it is referenced by multiple places.
 *  The function returns the raw pointer to the node to allow modification
 *  of the content.
 *
 * \code
 *
 *  MyCOWNodeRef ref, ref2;
 *  ref2 = ref;
 *  ref.CopyOnWrite()->value = new_value;
 *  assert(ref2->value == old_value);
 *  assert(ref->value == new_value);
 *
 * \endcode
 */
#define TVM_DEFINE_NODE_REF_COW(NodeName)                               \
  NodeName* CopyOnWrite() {                                             \
      CHECK(data_ != nullptr);                                          \
      if (!data_.unique())  {                                           \
        NodePtr<NodeName> n = make_node<NodeName>(*(operator->()));     \
        ObjectPtr<Object>(std::move(n)).swap(data_);                    \
      }                                                                 \
      return static_cast<NodeName*>(data_.get());                       \
    }

/*! \brief Macro to make it easy to define node ref type given node */
#define TVM_DEFINE_NODE_REF(TypeName, NodeName)                      \
  class TypeName : public ::tvm::NodeRef {                           \
   public:                                                           \
    TVM_DEFINE_NODE_REF_METHODS(TypeName, ::tvm::NodeRef, NodeName); \
  };                                                                 \

/*!
 * \brief Macro to make it easy to define node ref type that
 *  has a CopyOnWrite member function.
 */
#define TVM_DEFINE_COW_NODE_REF(TypeName, BaseType, NodeName)           \
  class TypeName : public BaseType {                                    \
   public:                                                              \
    TVM_DEFINE_NODE_REF_METHODS(TypeName, BaseType, NodeName);          \
    TVM_DEFINE_NODE_REF_COW(NodeName);                                  \
  };
}  // namespace tvm
#endif  // TVM_NODE_NODE_H_
