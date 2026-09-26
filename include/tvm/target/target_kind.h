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
 * \file tvm/target/target_kind.h
 * \brief Target kind registry
 */
#ifndef TVM_TARGET_TARGET_KIND_H_
#define TVM_TARGET_TARGET_KIND_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/config_schema.h>
#include <tvm/runtime/base.h>

#include <mutex>
#include <utility>
#include <vector>

namespace tvm {

class Target;

/*!
 * \brief Target canonicalizer applied on instantiation of a given TargetKind.
 *
 * \param target_json Target in JSON format to be transformed during canonicalization.
 * \return The transformed Target JSON object.
 */
using FTargetCanonicalizer =
    ffi::TypedFunction<ffi::Map<ffi::String, ffi::Any>(ffi::Map<ffi::String, ffi::Any>)>;

class TargetInternal;

/*! \brief Target kind, specifies the kind of the target */
class TargetKindNode : public ffi::Object {
 public:
  /*! \brief Name of the target kind */
  ffi::String name;
  /*! \brief Device type of target kind */
  int default_device_type;
  /*! \brief Default keys of the target */
  ffi::Array<ffi::String> default_keys;
  /*! \brief Function used to canonicalize a JSON target during creation */
  FTargetCanonicalizer target_canonicalizer;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TargetKindNode>()
        .def_ro("name", &TargetKindNode::name)
        .def_ro("default_device_type", &TargetKindNode::default_device_type,
                refl::AttachFieldFlag::SEqHashIgnore())
        .def_ro("default_keys", &TargetKindNode::default_keys,
                refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUniqueInstance;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("target.TargetKind", TargetKindNode, ffi::Object);

 private:
  /*! \brief ConfigSchema for validating and resolving target attributes */
  ir::ConfigSchema schema_;
  friend class TargetKindRegistry;
  friend class TargetKindDef;
  friend class TargetInternal;
};

/*!
 * \brief Managed reference class to TargetKindNode
 * \sa TargetKindNode
 */
class TargetKind : public ffi::ObjectRef {
 public:
  TargetKind() = default;
  explicit TargetKind(ffi::ObjectPtr<TargetKindNode> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*!
   * \brief Retrieve the TargetKind given its name
   * \param target_kind_name Name of the target kind
   * \return The canonical kind, or nullopt when the name is unknown.
   */
  TVM_DLL static ffi::Optional<TargetKind> Get(const ffi::String& target_kind_name);
  /*! \brief Mutable access to the container class  */
  TargetKindNode* operator->() { return static_cast<TargetKindNode*>(data_.get()); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TargetKind, ffi::ObjectRef, TargetKindNode);

 private:
  friend class TargetInternal;
};

/*! \brief Value used with --runtime in target specs to indicate the C++ runtime. */
static constexpr const char* kTvmRuntimeCpp = "c++";

/*! \brief Value used with --runtime in target specs to indicate the C runtime. */
static constexpr const char* kTvmRuntimeCrt = "c";

/*!
 * \brief Process-wide registry of canonical target kinds, keyed by name.
 */
class TargetKindRegistry {
 public:
  /*!
   * \brief Access the process-wide registry.
   * \return The singleton registry.
   */
  TVM_DLL static TargetKindRegistry* Global();
  /*!
   * \brief Register a name or return its existing canonical kind.
   * \param name Target kind name.
   * \return The canonical target kind for name. Standard target options are
   * declared when the name is first registered.
   */
  TVM_DLL TargetKind RegisterOrGet(const ffi::String& name);
  /*!
   * \brief Look up a registered kind.
   * \param name Target kind name.
   * \return The canonical kind, or nullopt when name is unknown.
   */
  TVM_DLL ffi::Optional<TargetKind> Get(const ffi::String& name);
  /*!
   * \brief List registered kind names.
   * \return Names of all registered target kinds.
   */
  TVM_DLL ffi::Array<ffi::String> ListTargetKinds();
  /*!
   * \brief List declared option names and types for a kind.
   * \param kind Registered target kind.
   * \return Map from option name to type string.
   */
  TVM_DLL ffi::Map<ffi::String, ffi::String> ListTargetKindOptions(const TargetKind& kind);

 private:
  std::mutex mutex_;
  ffi::Map<ffi::String, TargetKind> kinds_;
};

/*!
 * \brief Temporary fluent builder for registering a target kind.
 *
 * The registry retains the canonical kind after this builder is destroyed.
 * Group related definitions inside TVM_FFI_STATIC_INIT_BLOCK():
 * \code
 * TargetKindDef("llvm").set_default_device_type(kDLCPU)
 *     .set_default_keys({"cpu"}).def_option<ffi::String>("mcpu");
 * \endcode
 */
class TargetKindDef {
 public:
  /*!
   * \brief Register or retrieve the canonical kind with this name.
   * \param name Target kind name.
   */
  explicit TargetKindDef(const ffi::String& name)
      : kind_(TargetKindRegistry::Global()->RegisterOrGet(name)) {}
  /*!
   * \brief Set the default DLPack device type.
   * \param device_type DLPack device type.
   * \return This builder for chaining.
   */
  TargetKindDef& set_default_device_type(int device_type) {
    kind_->default_device_type = device_type;
    return *this;
  }
  /*!
   * \brief Set default target keys.
   * \param keys Default keys in priority order.
   * \return This builder for chaining.
   */
  TargetKindDef& set_default_keys(std::vector<ffi::String> keys) {
    kind_->default_keys = keys;
    return *this;
  }
  /*!
   * \brief Set the canonicalizer used when constructing targets of this kind.
   * \param canonicalizer Function from a validated config map to its canonical form.
   * \return This builder for chaining. Also updates the kind's ConfigSchema.
   */
  TargetKindDef& set_target_canonicalizer(FTargetCanonicalizer canonicalizer) {
    kind_->target_canonicalizer = canonicalizer;
    kind_->schema_.set_canonicalizer(canonicalizer);
    return *this;
  }
  /*!
   * \brief Declare a typed target option in the kind's ConfigSchema.
   * \tparam ValueType Canonical option value type.
   * \tparam Traits Optional metadata or validator trait types.
   * \param key Option name.
   * \param traits Optional traits such as a default value or validator.
   * \return This builder for chaining. Duplicate option names raise ValueError.
   */
  template <typename ValueType, typename... Traits>
  TargetKindDef& def_option(const ffi::String& key, Traits&&... traits) {
    kind_->schema_.def_option<ValueType>(key, std::forward<Traits>(traits)...);
    return *this;
  }

 private:
  TargetKind kind_;
};

}  // namespace tvm

#endif  // TVM_TARGET_TARGET_KIND_H_
