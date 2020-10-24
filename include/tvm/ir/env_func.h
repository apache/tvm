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
 * \file tvm/ir/env_func.h
 * \brief Serializable global function used in IR.
 */
#ifndef TVM_IR_ENV_FUNC_H_
#define TVM_IR_ENV_FUNC_H_

#include <tvm/node/reflection.h>

#include <string>
#include <utility>

namespace tvm {
/*!
 * \brief A serializable function backed by TVM's global environment.
 *
 * This is a wrapper to enable serializable global PackedFunc.
 * An EnvFunc is saved by its name in the global registry
 * under the assumption that the same function is registered during load.
 * \sa EnvFunc
 */
class EnvFuncNode : public Object {
 public:
  /*! \brief Unique name of the global function */
  String name;
  /*! \brief The internal packed function */
  runtime::PackedFunc func;
  /*! \brief constructor */
  EnvFuncNode() {}

  void VisitAttrs(AttrVisitor* v) { v->Visit("name", &name); }

  bool SEqualReduce(const EnvFuncNode* other, SEqualReducer equal) const {
    // name uniquely identifies the env function.
    return name == other->name;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    // Name uniquely identifies the env function.
    hash_reduce(name);
  }

  static constexpr const char* _type_key = "EnvFunc";
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(EnvFuncNode, Object);
};

/*!
 * \brief Managed reference to EnvFuncNode.
 * \sa EnvFuncNode
 */
class EnvFunc : public ObjectRef {
 public:
  EnvFunc() {}
  explicit EnvFunc(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*! \return The internal global function pointer */
  const EnvFuncNode* operator->() const { return static_cast<const EnvFuncNode*>(get()); }
  /*!
   * \brief Invoke the function.
   * \param args The arguments
   * \returns The return value.
   */
  template <typename... Args>
  runtime::TVMRetValue operator()(Args&&... args) const {
    const EnvFuncNode* n = operator->();
    ICHECK(n != nullptr);
    return n->func(std::forward<Args>(args)...);
  }
  /*!
   * \brief Get a global function based on the name.
   * \param name The name of the global function.
   * \return The created global function.
   * \note The function can be unique
   */
  TVM_DLL static EnvFunc Get(const String& name);
  /*! \brief specify container node */
  using ContainerType = EnvFuncNode;
};

/*!
 * \brief Please refer to \ref TypedEnvFuncAnchor "TypedEnvFunc<R(Args..)>"
 */
template <typename FType>
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
template <typename R, typename... Args>
class TypedEnvFunc<R(Args...)> : public ObjectRef {
 public:
  /*! \brief short hand for this function type */
  using TSelf = TypedEnvFunc<R(Args...)>;
  TypedEnvFunc() {}
  explicit TypedEnvFunc(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief Assign global function to a TypedEnvFunc
   * \param other Another global function.
   * \return reference to self.
   */
  TSelf& operator=(const EnvFunc& other) {
    ObjectRef::operator=(other);
    return *this;
  }
  /*! \return The internal global function pointer */
  const EnvFuncNode* operator->() const { return static_cast<const EnvFuncNode*>(get()); }
  /*!
   * \brief Invoke the function.
   * \param args The arguments
   * \returns The return value.
   */
  R operator()(Args... args) const {
    const EnvFuncNode* n = operator->();
    ICHECK(n != nullptr);
    return runtime::detail::typed_packed_call_dispatcher<R>::run(n->func,
                                                                 std::forward<Args>(args)...);
  }
  /*! \brief specify container node */
  using ContainerType = EnvFuncNode;
};

}  // namespace tvm
#endif  // TVM_IR_ENV_FUNC_H_
