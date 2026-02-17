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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/node/node.h>
#include <tvm/runtime/device_api.h>
#include <tvm/support/with.h>
#include <tvm/target/target_kind.h>

#include <string>

namespace tvm {

class TargetInternal;
class Target;

/*!
 * \brief Compilation target.
 * \sa Target
 */
class TargetNode : public Object {
 public:
  /*! \brief The kind of the target device */
  TargetKind kind;
  /*! \brief Target host information, must be Target type */
  ffi::Optional<ObjectRef> host;
  /*! \brief Tag of the target, can be empty */
  ffi::String tag;
  /*! \brief Keys for this target */
  ffi::Array<ffi::String> keys;
  /*! \brief Collection of attributes (includes feature.* keys set by canonicalizer) */
  ffi::Map<ffi::String, Any> attrs;

  /*!
   * \brief The JSON string representation of the target
   * \return JSON string of the target configuration (e.g. {"kind": "llvm", "mcpu": "cortex-a53"})
   */
  TVM_DLL const std::string& str() const;
  /*! \return Export target to JSON-like configuration */
  TVM_DLL ffi::Map<ffi::String, ffi::Any> ToConfig() const;
  /*! \return The ffi::Optional<Target> typed target host of the TargetNode */
  TVM_DLL ffi::Optional<Target> GetHost() const;
  /*! \return The device type for this target */
  TVM_DLL int GetTargetDeviceType() const;

  /*!
   * \brief Check if the target contains a key
   *
   * \param query_key The string name of the key to be checked
   *
   * \return True if the target's `TargetNode::keys` contains the
   * specified key, False otherwise.
   */
  TVM_DLL bool HasKey(const std::string& query_key) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TargetNode>()
        .def_ro("kind", &TargetNode::kind)
        .def_ro("tag", &TargetNode::tag)
        .def_ro("keys", &TargetNode::keys)
        .def_ro("attrs", &TargetNode::attrs)
        .def_ro("host", &TargetNode::host);
  }

  /*!
   * \brief Get an entry from attrs of the target
   * \tparam TObjectRef Type of the attribute
   * \param attr_key The name of the attribute key
   * \param default_value The value returned if the key is not present
   * \return An optional, std::nullopt if not found, otherwise the value found
   */
  template <typename TObjectRef>
  ffi::Optional<TObjectRef> GetAttr(
      const std::string& attr_key,
      ffi::Optional<TObjectRef> default_value = ffi::Optional<TObjectRef>(std::nullopt)) const {
    auto it = attrs.find(attr_key);
    if (it != attrs.end()) {
      return Downcast<ffi::Optional<TObjectRef>>((*it).second);
    } else {
      return default_value;
    }
  }
  /*!
   * \brief Get an entry from attrs of the target
   * \tparam TObjectRef Type of the attribute
   * \param attr_key The name of the attribute key
   * \param default_value The value returned if the key is not present
   * \return An optional, std::nullopt if not found, otherwise the value found
   */
  template <typename TObjectRef>
  ffi::Optional<TObjectRef> GetAttr(const std::string& attr_key, TObjectRef default_value) const {
    return GetAttr<TObjectRef>(attr_key, ffi::Optional<TObjectRef>(default_value));
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("target.Target", TargetNode, Object);

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
   * \param tag_or_config_or_target_str the string to parse for target
   */
  TVM_DLL explicit Target(const ffi::String& tag_or_config_or_target_str);
  /*!
   * \brief Construct a Target using a JSON-like configuration
   * \param config The JSON-like configuration for target
   */
  TVM_DLL explicit Target(const ffi::Map<ffi::String, ffi::Any>& config);
  /*!
   * \brief Get the current target context from thread local storage.
   * \param allow_not_defined If the context stack is empty and this is set to true, an
   *   undefined Target will be returned. Otherwise, an empty context stack will cause a
   *   runtime error.
   * \return The target that is the current context. The target may not be defined if
   * allow_not_defined is true.
   */
  TVM_DLL static tvm::Target Current(bool allow_not_defined = true);
  /*!
   * \brief Construct a Target given target and host
   * \param target The Target typed object with host field undefined for target
   * \param host The Target typed object for target host
   */
  TVM_DLL explicit Target(Target target, Target host);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Target, ObjectRef, TargetNode);

  static Target WithHost(const Target& target, const Target& host);

  /*! \return The target with the host stripped out */
  Target WithoutHost() const;

 private:
  Target(TargetKind kind, ffi::Optional<ObjectRef> host, ffi::String tag,
         ffi::Array<ffi::String> keys, ffi::Map<ffi::String, ffi::Any> attrs);

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

/*!
 * \brief Check and update host field of the given legacy target and target host pair.
 *  Note that this function is for legacy target api compatibility issue only, not
 *  recommended for other use.
 * \param target The pointer to a Target typed object with host field to be updated
 * \param host The pointer to a Target typed object for target host to be updated
 */
void CheckAndUpdateHostConsistency(Target* target, Target* host);

}  // namespace tvm
#endif  // TVM_TARGET_TARGET_H_
