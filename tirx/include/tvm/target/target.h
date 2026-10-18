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
#include <tvm/ir/with_context.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/target_kind.h>

#include <string>

namespace tvm {

namespace backend::attr {
/*! \brief Optional backend configuration on a Target, distinct from serialized IR/runtime JSON. */
inline constexpr ffi::reflection::AttrKey<ffi::Map<ffi::String, ffi::Any>> kBackendConfig{
    "backend_config"};
}  // namespace backend::attr

class TargetInternal;
class Target;

/*!
 * \brief Compilation target.
 * \sa Target
 */
class TargetObj : public ffi::Object {
 public:
  explicit TargetObj(TargetKind kind) : kind(std::move(kind)) {}
  explicit TargetObj(ffi::UnsafeInit tag) : kind(tag) {}
  /*! \brief The kind of the target device */
  TargetKind kind;
  /*! \brief Target host information, must be Target type */
  ffi::Optional<ffi::ObjectRef> host;
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
  TVM_TIRX_DLL const std::string& str() const;
  /*! \return Export target to JSON-like configuration */
  TVM_TIRX_DLL ffi::Map<ffi::String, ffi::Any> ToConfig() const;
  /*! \return The ffi::Optional<Target> typed target host of the TargetObj */
  TVM_TIRX_DLL ffi::Optional<Target> GetHost() const;
  /*! \return The device type for this target */
  TVM_TIRX_DLL int GetTargetDeviceType() const;

  /*!
   * \brief Check if the target contains a key
   *
   * \param query_key The string name of the key to be checked
   *
   * \return True if the target's `TargetObj::keys` contains the
   * specified key, False otherwise.
   */
  TVM_TIRX_DLL bool HasKey(const std::string& query_key) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TargetObj>()
        .def_ro("kind", &TargetObj::kind)
        .def_ro("tag", &TargetObj::tag)
        .def_ro("keys", &TargetObj::keys)
        .def_ro("attrs", &TargetObj::attrs)
        .def_ro("host", &TargetObj::host);
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
      return (*it).second.as_or_throw<ffi::Optional<TObjectRef>>();
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

  /*! \brief Get an optional Target attribute using the value type declared by its key. */
  template <typename T>
  ffi::Optional<T> GetAttr(ffi::reflection::AttrKey<T> key,
                           ffi::Optional<typename ffi::reflection::AttrKey<T>::ValueType>
                               default_value = std::nullopt) const {
    return GetAttr<T>(key.name, std::move(default_value));
  }

  /*! \brief Use a default value when the typed Target attribute is absent. */
  template <typename T>
  ffi::Optional<T> GetAttr(ffi::reflection::AttrKey<T> key,
                           typename ffi::reflection::AttrKey<T>::ValueType default_value) const {
    return GetAttr(key, ffi::Optional<T>(std::move(default_value)));
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("target.Target", TargetObj, ffi::Object);

 private:
  /*! \brief Internal string repr. */
  mutable std::string str_repr_;

  friend class TargetInternal;
};

/*!
 * \brief Managed reference class to TargetObj.
 * \sa TargetObj
 */
class Target : public ffi::ObjectRef {
 public:
  /*!
   * \brief Construct a Target given a string
   * \param tag_or_config_or_target_str the string to parse for target
   */
  TVM_TIRX_DLL explicit Target(const ffi::String& tag_or_config_or_target_str);
  /*!
   * \brief Construct a Target using a JSON-like configuration
   * \param config The JSON-like configuration for target
   */
  TVM_TIRX_DLL explicit Target(const ffi::Map<ffi::String, ffi::Any>& config);
  /*!
   * \brief Get the current target context from thread local storage.
   * \param allow_not_defined If the context stack is empty and this is set to true, std::nullopt
   *   will be returned. Otherwise, an empty context stack will cause a
   *   runtime error.
   * \return The target that is the current context. The result may be absent if
   * allow_not_defined is true.
   */
  TVM_TIRX_DLL static ffi::Optional<Target> Current(bool allow_not_defined = true);
  /*!
   * \brief Construct a Target given target and host
   * \param target The Target typed object with host field undefined for target
   * \param host The Target typed object for target host
   */
  TVM_TIRX_DLL explicit Target(Target target, ffi::Optional<Target> host);
  explicit Target(ffi::UnsafeInit tag, ffi::ObjectPtr<TargetObj> n) : ffi::ObjectRef(tag) {
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Target, ffi::ObjectRef, TargetObj);

  static Target WithHost(const Target& target, const ffi::Optional<Target>& host);

  /*! \return The target with the host stripped out */
  Target WithoutHost() const;

 private:
  // enable with syntax.
  friend class TargetInternal;
  friend class With<Target>;
  /*!
   * \brief Push a new target context onto the thread local stack.
   *  The Target on top of the stack is returned by Target::Current().
   */
  TVM_TIRX_DLL void EnterWithScope();
  /*!
   * \brief Pop a target off the thread local context stack,
   *  restoring the previous target as the current context.
   */
  TVM_TIRX_DLL void ExitWithScope();
};

}  // namespace tvm
#endif  // TVM_TARGET_TARGET_H_
