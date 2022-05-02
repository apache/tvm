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
 * \file tvm/relay/runtime.h
 * \brief Object representation of Runtime configuration and registry
 */
#ifndef TVM_RELAY_RUNTIME_H_
#define TVM_RELAY_RUNTIME_H_

#include <dmlc/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>
#include <tvm/ir/type_relation.h>
#include <tvm/node/attr_registry_map.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {

template <typename, typename>
class AttrRegistry;

namespace relay {

/*! \brief Value used with Runtime::name to indicate the C++ runtime. */
static constexpr const char* kTvmRuntimeCpp = "cpp";

/*! \brief Value used with Runtime::name to indicate the C runtime. */
static constexpr const char* kTvmRuntimeCrt = "crt";

/*!
 * \brief Runtime information.
 *
 * This data structure stores the meta-data
 * about Runtimes which can be used to pass around information.
 *
 * \sa Runtime
 */
class RuntimeNode : public Object {
 public:
  /*! \brief name of the Runtime */
  String name;
  /* \brief Additional attributes storing meta-data about the Runtime. */
  DictAttrs attrs;

  /*!
   * \brief Get an attribute.
   *
   * \param attr_key The attribute key.
   * \param default_value The default value if the key does not exist, defaults to nullptr.
   *
   * \return The result
   *
   * \tparam TObjectRef the expected object type.
   * \throw Error if the key exists but the value does not match TObjectRef
   *
   * \code
   *
   *  void GetAttrExample(const Runtime& runtime) {
   *    auto value = runtime->GetAttr<Integer>("AttrKey", 0);
   *  }
   *
   * \endcode
   */
  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(
      const std::string& attr_key,
      Optional<TObjectRef> default_value = Optional<TObjectRef>(nullptr)) const {
    return attrs.GetAttr(attr_key, default_value);
  }
  // variant that uses TObjectRef to enable implicit conversion to default value.
  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(const std::string& attr_key, TObjectRef default_value) const {
    return GetAttr<TObjectRef>(attr_key, Optional<TObjectRef>(default_value));
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("attrs", &attrs);
  }

  bool SEqualReduce(const RuntimeNode* other, SEqualReducer equal) const {
    return name == other->name && equal.DefEqual(attrs, other->attrs);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(attrs);
  }

  static constexpr const char* _type_key = "Runtime";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(RuntimeNode, Object);
};

/*!
 * \brief Managed reference class to RuntimeNode.
 * \sa RuntimeNode
 */
class Runtime : public ObjectRef {
 public:
  Runtime() = default;

  /*!
   * \brief Create a new Runtime object using the registry
   * \throws Error if name is not registered
   * \param name The name of the Runtime.
   * \param attrs Attributes for the Runtime.
   * \return the new Runtime object.
   */
  TVM_DLL static Runtime Create(String name, Map<String, ObjectRef> attrs = {});

  /*!
   * \brief List all registered Runtimes
   * \return the list of Runtimes
   */
  TVM_DLL static Array<String> ListRuntimes();

  /*!
   * \brief List all options for a specific Runtime
   * \param name The name of the Runtime
   * \return Map of option name to type
   */
  TVM_DLL static Map<String, String> ListRuntimeOptions(const String& name);

  /*! \brief specify container node */
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Runtime, ObjectRef, RuntimeNode);

 private:
  /*!
   * \brief Private Constructor
   * \param name The Runtime name
   * \param attrs Attributes to apply to this Runtime node
   */
  TVM_DLL Runtime(String name, DictAttrs attrs) {
    auto n = make_object<RuntimeNode>();
    n->name = std::move(name);
    n->attrs = std::move(attrs);
    data_ = std::move(n);
  }
};

/*!
 * \brief Helper structure to register Runtimes
 * \sa TVM_REGISTER_Runtime
 */
class RuntimeRegEntry {
 public:
  /*!
   * \brief Register a valid configuration option and its ValueType for validation
   * \param key The configuration key
   * \tparam ValueType The value type to be registered
   */
  template <typename ValueType>
  inline RuntimeRegEntry& add_attr_option(const String& key);

  /*!
   * \brief Register a valid configuration option and its ValueType for validation
   * \param key The configuration key
   * \param default_value The default value of the key
   * \tparam ValueType The value type to be registered
   */
  template <typename ValueType>
  inline RuntimeRegEntry& add_attr_option(const String& key, ObjectRef default_value);

  /*!
   * \brief Register or get a new entry.
   * \param name The name of the operator.
   * \return the corresponding entry.
   */
  TVM_DLL static RuntimeRegEntry& RegisterOrGet(const String& name);

 private:
  /*! \brief Internal storage of value types */
  struct ValueTypeInfo {
    std::string type_key;
    uint32_t type_index;
  };
  std::unordered_map<std::string, ValueTypeInfo> key2vtype_;
  /*! \brief A hash table that stores the default value of each attr */
  std::unordered_map<String, ObjectRef> key2default_;

  /*! \brief Index used for internal lookup of attribute registry */
  uint32_t index_;

  // the name
  std::string name;

  /*! \brief Return the index stored in attr registry */
  uint32_t AttrRegistryIndex() const { return index_; }
  /*! \brief Return the name stored in attr registry */
  String AttrRegistryName() const { return name; }

  /*! \brief private constructor */
  explicit RuntimeRegEntry(uint32_t reg_index) : index_(reg_index) {}

  // friend class
  template <typename>
  friend class AttrRegistryMapContainerMap;
  template <typename, typename>
  friend class tvm::AttrRegistry;
  friend class Runtime;
};

template <typename ValueType>
inline RuntimeRegEntry& RuntimeRegEntry::add_attr_option(const String& key) {
  ICHECK(!key2vtype_.count(key)) << "AttributeError: add_attr_option failed because '" << key
                                 << "' has been set once";

  using ValueNodeType = typename ValueType::ContainerType;
  // NOTE: we could further update the function later.
  uint32_t value_type_index = ValueNodeType::_GetOrAllocRuntimeTypeIndex();

  ValueTypeInfo info;
  info.type_index = value_type_index;
  info.type_key = runtime::Object::TypeIndex2Key(value_type_index);
  key2vtype_[key] = info;
  return *this;
}

template <typename ValueType>
inline RuntimeRegEntry& RuntimeRegEntry::add_attr_option(const String& key,
                                                         ObjectRef default_value) {
  add_attr_option<ValueType>(key);
  key2default_[key] = default_value;
  return *this;
}

// internal macros to make Runtime entries
#define TVM_RUNTIME_REGISTER_VAR_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::relay::RuntimeRegEntry& __make_##Runtime

/*!
 * \def TVM_REGISTER_RUNTIME
 * \brief Register a new Runtime, or set attribute of the corresponding Runtime.
 *
 * \param RuntimeName The name of registry
 *
 * \code
 *
 *  TVM_REGISTER_RUNTIME("c")
 *  .add_attr_option<String>("my_option");
 *  .add_attr_option<String>("my_option_default", String("default"));
 *
 * \endcode
 */
#define TVM_REGISTER_RUNTIME(RuntimeName)                     \
  TVM_STR_CONCAT(TVM_RUNTIME_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::relay::RuntimeRegEntry::RegisterOrGet(RuntimeName)
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_RUNTIME_H_
