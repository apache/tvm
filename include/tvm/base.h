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
 * \file tvm/base.h
 * \brief Defines the base data structure
 */
#ifndef TVM_BASE_H_
#define TVM_BASE_H_

#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <tvm/node/node.h>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include "runtime/registry.h"

namespace tvm {

using ::tvm::Node;
using ::tvm::NodeRef;
using ::tvm::AttrVisitor;

/*!
 * \brief Macro to define common node ref methods.
 * \param TypeName The name of the NodeRef.
 * \param BaseTypeName The Base type.
 * \param NodeName The node container type.
 */
#define TVM_DEFINE_NODE_REF_METHODS(TypeName, BaseTypeName, NodeName)   \
  TypeName() {}                                                         \
  explicit TypeName(::tvm::NodePtr<::tvm::Node> n) : BaseTypeName(n) {} \
  const NodeName* operator->() const {                                  \
    return static_cast<const NodeName*>(node_.get());                   \
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
      CHECK(node_ != nullptr);                                          \
      if (!node_.unique())  {                                           \
        NodePtr<NodeName> n = make_node<NodeName>(*(operator->()));     \
        NodePtr<Node>(std::move(n)).swap(node_);                        \
      }                                                                 \
      return static_cast<NodeName*>(node_.get());                       \
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

/*!
 * \brief RAII wrapper function to enter and exit a context object
 *        similar to python's with syntax.
 *
 * \code
 * // context class
 * class MyContext {
 *  private:
 *    friend class With<MyContext>;
      MyContext(arguments);
 *    void EnterWithScope();
 *    void ExitWithScope();
 * };
 *
 * {
 *   With<MyContext> scope(arguments);
 *   // effect take place.
 * }
 * \endcode
 *
 * \tparam ContextType Type of the context object.
 */
template<typename ContextType>
class With {
 public:
  /*!
   * \brief constructor.
   *  Enter the scope of the context.
   */
  template<typename ...Args>
  explicit With(Args&& ...args)
      : ctx_(std::forward<Args>(args)...) {
    ctx_.EnterWithScope();
  }
  /*! \brief destructor, leaves the scope of the context. */
  ~With() DMLC_THROW_EXCEPTION {
    ctx_.ExitWithScope();
  }

 private:
  /*! \brief internal context type. */
  ContextType ctx_;
};

/*!
 * \brief save the node as well as all the node it depends on as json.
 *  This can be used to serialize any TVM object
 *
 * \return the string representation of the node.
 */
std::string SaveJSON(const NodeRef& node);

/*!
 * \brief Internal implementation of LoadJSON
 * Load tvm Node object from json and return a shared_ptr of Node.
 * \param json_str The json string to load from.
 *
 * \return The shared_ptr of the Node.
 */
NodePtr<Node> LoadJSON_(std::string json_str);

/*!
 * \brief Load the node from json string.
 *  This can be used to deserialize any TVM object.
 *
 * \param json_str The json string to load from.
 *
 * \tparam NodeType the nodetype
 *
 * \code
 *  Expr e = LoadJSON<Expr>(json_str);
 * \endcode
 */
template<typename NodeType,
         typename = typename std::enable_if<std::is_base_of<NodeRef, NodeType>::value>::type >
inline NodeType LoadJSON(const std::string& json_str) {
  return NodeType(LoadJSON_(json_str));
}

/*!
 * \brief Registry entry for NodeFactory.
 *
 *  There are two types of Nodes that can be serialized.
 *  The normal node requires a registration a creator function that
 *  constructs an empty Node of the corresponding type.
 *
 *  The global singleton(e.g. global operator) where only global_key need to be serialized,
 *  in this case, FGlobalKey need to be defined.
 */
struct NodeFactoryReg {
  /*!
   * \brief creator function.
   * \param global_key Key that identifies a global single object.
   *        If this is not empty then FGlobalKey
   * \return The created function.
   */
  using FCreate = std::function<NodePtr<Node>(const std::string& global_key)>;
  /*!
   * \brief Global key function, only needed by global objects.
   * \param node The node pointer.
   * \return node The global key to the node.
   */
  using FGlobalKey = std::function<std::string(const Node* node)>;
  /*! \brief registered name */
  std::string name;
  /*!
   * \brief The creator function
   */
  FCreate fcreator = nullptr;
  /*!
   * \brief The global key function.
   */
  FGlobalKey fglobal_key = nullptr;
  // setter of creator
  NodeFactoryReg& set_creator(FCreate f) {  // NOLINT(*)
    this->fcreator = f;
    return *this;
  }
  // setter of creator
  NodeFactoryReg& set_global_key(FGlobalKey f) {  // NOLINT(*)
    this->fglobal_key = f;
    return *this;
  }
  // global registry singleton
  TVM_DLL static ::dmlc::Registry<::tvm::NodeFactoryReg> *Registry();
};

/*!
 * \brief Register a Node type
 * \note This is necessary to enable serialization of the Node.
 */
#define TVM_REGISTER_NODE_TYPE(TypeName)                                \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::NodeFactoryReg & __make_Node ## _ ## TypeName ## __ = \
      ::tvm::NodeFactoryReg::Registry()->__REGISTER__(TypeName::_type_key) \
      .set_creator([](const std::string&) { return ::tvm::make_node<TypeName>(); })


#define TVM_STRINGIZE_DETAIL(x) #x
#define TVM_STRINGIZE(x) TVM_STRINGIZE_DETAIL(x)
#define TVM_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" TVM_STRINGIZE(__LINE__))
/*!
 * \brief Macro to include current line as string
 */
#define TVM_ADD_FILELINE "\n\nDefined in " __FILE__ ":L" TVM_STRINGIZE(__LINE__)


}  // namespace tvm
#endif  // TVM_BASE_H_
