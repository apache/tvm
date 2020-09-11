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
 * \file tvm/target/target.h
 * \brief Compilation target object.
 */
#ifndef TVM_TARGET_TARGET_H_
#define TVM_TARGET_TARGET_H_

#include <tvm/node/node.h>
#include <tvm/support/with.h>
#include <tvm/target/target_kind.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace tvm {

class TargetInternal;

/*!
 * \brief Compilation target.
 * \sa Target
 */
class TargetNode : public Object {
 public:
  /*! \brief The kind of the target device */
  TargetKind kind;
  /*! \brief Tag of the the target, can be empty */
  String tag;
  /*! \brief Keys for this target */
  Array<String> keys;
  /*! \brief Collection of attributes */
  Map<String, ObjectRef> attrs;
  /*!
   * \brief The raw string representation of the target
   * \return the full device string to pass to codegen::Build
   * \note It will be deprecated after the Target RFC is fully landed.
   */
  TVM_DLL const std::string& str() const;
  /*! \return Export target to JSON-like configuration */
  TVM_DLL Map<String, ObjectRef> Export() const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("kind", &kind);
    v->Visit("tag", &tag);
    v->Visit("keys", &keys);
    v->Visit("attrs", &attrs);
  }

  /*!
   * \brief Get an entry from attrs of the target
   * \tparam TObjectRef Type of the attribute
   * \param attr_key The name of the attribute key
   * \param default_value The value returned if the key is not present
   * \return An optional, NullOpt if not found, otherwise the value found
   */
  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(
      const std::string& attr_key,
      Optional<TObjectRef> default_value = Optional<TObjectRef>(nullptr)) const {
    static_assert(std::is_base_of<ObjectRef, TObjectRef>::value,
                  "Can only call GetAttr with ObjectRef types.");
    auto it = attrs.find(attr_key);
    if (it != attrs.end()) {
      return Downcast<Optional<TObjectRef>>((*it).second);
    } else {
      return default_value;
    }
  }
  /*!
   * \brief Get an entry from attrs of the target
   * \tparam TObjectRef Type of the attribute
   * \param attr_key The name of the attribute key
   * \param default_value The value returned if the key is not present
   * \return An optional, NullOpt if not found, otherwise the value found
   */
  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(const std::string& attr_key, TObjectRef default_value) const {
    return GetAttr<TObjectRef>(attr_key, Optional<TObjectRef>(default_value));
  }
  /*! \brief Get the keys for this target as a vector of string */
  TVM_DLL std::vector<std::string> GetKeys() const;
  /*! \brief Get the keys for this target as an unordered_set of string */
  TVM_DLL std::unordered_set<std::string> GetLibs() const;

  static constexpr const char* _type_key = "Target";
  TVM_DECLARE_FINAL_OBJECT_INFO(TargetNode, Object);

 private:
  /*! \brief Internal string repr. */
  mutable std::string str_repr_;

  friend class TargetInternal;
};

/*!
 * \brief Managed reference class to TargetNode.
 * \sa TargetNode
 */
class Target : public ObjectRef {
 public:
  /*! \brief Construct a null Target */
  TVM_DLL explicit Target(std::nullptr_t) { data_ = nullptr; }
  /*!
   * \brief Construct a Target given a string
   * \param tag_or_config_or_target_str the string to parse
   */
  TVM_DLL explicit Target(const String& tag_or_config_or_target_str);
  /*!
   * \brief Construct a Target using a JSON-like configuration
   * \param config The JSON-like configuration
   */
  TVM_DLL explicit Target(const Map<String, ObjectRef>& config);
  /*!
   * \brief Get the current target context from thread local storage.
   * \param allow_not_defined If the context stack is empty and this is set to true, an
   *   undefined Target will be returned. Otherwise, an empty context stack will cause a
   *   runtime error.
   * \return The target that is the current context. The target may not be defined if
   * allow_not_defined is true.
   */
  TVM_DLL static tvm::Target Current(bool allow_not_defined = true);

  TVM_DEFINE_OBJECT_REF_METHODS(Target, ObjectRef, TargetNode);

 private:
  // enable with syntax.
  friend class TargetInternal;
  friend class With<Target>;
  /*!
   * \brief Push a new target context onto the thread local stack.
   *  The Target on top of the stack is used to determine which
   *  specialization to use when invoking a GenericFunc.
   */
  TVM_DLL void EnterWithScope();
  /*!
   * \brief Pop a target off the thread local context stack,
   *  restoring the previous target as the current context.
   */
  TVM_DLL void ExitWithScope();
};

}  // namespace tvm
#endif  // TVM_TARGET_TARGET_H_
