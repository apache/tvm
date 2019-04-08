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
 * \file tvm/api_registry.h
 * \brief This file contains utilities related to
 *  the TVM's global function registry.
 */
#ifndef TVM_API_REGISTRY_H_
#define TVM_API_REGISTRY_H_

#include <string>
#include <utility>
#include "base.h"
#include "packed_func_ext.h"
#include "runtime/registry.h"

namespace tvm {
/*!
 * \brief Register an API function globally.
 * It simply redirects to TVM_REGISTER_GLOBAL
 *
 * \code
 *   TVM_REGISTER_API(MyPrint)
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *     // my code.
 *   });
 * \endcode
 */
#define TVM_REGISTER_API(OpName) TVM_REGISTER_GLOBAL(OpName)

/*!
 * \brief Node container of EnvFunc
 * \sa EnvFunc
 */
class EnvFuncNode : public Node {
 public:
  /*! \brief Unique name of the global function */
  std::string name;
  /*! \brief The internal packed function */
  PackedFunc func;
  /*! \brief constructor */
  EnvFuncNode() {}

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "EnvFunc";
  TVM_DECLARE_NODE_TYPE_INFO(EnvFuncNode, Node);
};

/*!
 * \brief A serializable function backed by TVM's global environment.
 *
 * This is a wrapper to enable serializable global PackedFunc.
 * An EnvFunc is saved by its name in the global registry
 * under the assumption that the same function is registered during load.
 */
class EnvFunc : public NodeRef {
 public:
  EnvFunc() {}
  explicit EnvFunc(NodePtr<Node> n) : NodeRef(n) {}
  /*! \return The internal global function pointer */
  const EnvFuncNode* operator->() const {
    return static_cast<EnvFuncNode*>(node_.get());
  }
  /*!
   * \brief Invoke the function.
   * \param args The arguments
   * \returns The return value.
   */
  template<typename... Args>
  runtime::TVMRetValue operator()(Args&&... args) const {
    const EnvFuncNode* n = operator->();
    CHECK(n != nullptr);
    return n->func(std::forward<Args>(args)...);
  }
  /*!
   * \brief Get a global function based on the name.
   * \param name The name of the global function.
   * \return The created global function.
   * \note The function can be unique
   */
  TVM_DLL static EnvFunc Get(const std::string& name);
  /*! \brief specify container node */
  using ContainerType = EnvFuncNode;
};

/*!
 * \brief Please refer to \ref TypedEnvFuncAnchor "TypedEnvFunc<R(Args..)>"
 */
template<typename FType>
class TypedEnvFunc;

/*!
 * \anchor TypedEnvFuncAnchor
 * \brief A typed version of EnvFunc.
 * It is backed by a GlobalFuncNode internally.
 *
 * \tparam R The return value of the function.
 * \tparam Args The argument signature of the function.
 * \sa EnvFunc
 */
template<typename R, typename... Args>
class TypedEnvFunc<R(Args...)> : public NodeRef {
 public:
  /*! \brief short hand for this function type */
  using TSelf = TypedEnvFunc<R(Args...)>;
  TypedEnvFunc() {}
  explicit TypedEnvFunc(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief Assign global function to a TypedEnvFunc
   * \param other Another global function.
   * \return reference to self.
   */
  TSelf& operator=(const EnvFunc& other) {
    this->node_ = other.node_;
    return *this;
  }
  /*! \return The internal global function pointer */
  const EnvFuncNode* operator->() const {
    return static_cast<EnvFuncNode*>(node_.get());
  }
  /*!
   * \brief Invoke the function.
   * \param args The arguments
   * \returns The return value.
   */
  R operator()(Args... args) const {
    const EnvFuncNode* n = operator->();
    CHECK(n != nullptr);
    return runtime::detail::typed_packed_call_dispatcher<R>
        ::run(n->func, std::forward<Args>(args)...);
  }
  /*! \brief specify container node */
  using ContainerType = EnvFuncNode;
};

}  // namespace tvm
#endif  // TVM_API_REGISTRY_H_
