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
 * \file tvm/target/target_id.h
 * \brief Target id registry
 */
#ifndef TVM_TARGET_TARGET_ID_H_
#define TVM_TARGET_TARGET_ID_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/node/attr_registry_map.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/support/with.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace detail {
template <typename, typename, typename>
struct ValueTypeInfoMaker;
}

class Target;

/*! \brief Perform schema validation */
TVM_DLL void TargetValidateSchema(const Map<String, ObjectRef>& config);

template <typename>
class TargetIdAttrMap;

/*! \brief Target Id, specifies the kind of the target */
class TargetIdNode : public Object {
 public:
  /*! \brief Name of the target id */
  String name;
  /*! \brief Device type of target id */
  int device_type;
  /*! \brief Default keys of the target */
  Array<String> default_keys;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("device_type", &device_type);
    v->Visit("default_keys", &default_keys);
  }

  Map<String, ObjectRef> ParseAttrsFromRaw(const std::vector<std::string>& options) const;

  Optional<String> StringifyAttrsToRaw(const Map<String, ObjectRef>& attrs) const;

  static constexpr const char* _type_key = "TargetId";
  TVM_DECLARE_FINAL_OBJECT_INFO(TargetIdNode, Object);

 private:
  /*! \brief Stores the required type_key and type_index of a specific attr of a target */
  struct ValueTypeInfo {
    String type_key;
    uint32_t type_index;
    std::unique_ptr<ValueTypeInfo> key;
    std::unique_ptr<ValueTypeInfo> val;
  };

  uint32_t AttrRegistryIndex() const { return index_; }
  String AttrRegistryName() const { return name; }
  /*! \brief Perform schema validation */
  void ValidateSchema(const Map<String, ObjectRef>& config) const;
  /*! \brief Verify if the obj is consistent with the type info */
  void VerifyTypeInfo(const ObjectRef& obj, const TargetIdNode::ValueTypeInfo& info) const;
  /*! \brief A hash table that stores the type information of each attr of the target key */
  std::unordered_map<String, ValueTypeInfo> key2vtype_;
  /*! \brief A hash table that stores the default value of each attr of the target key */
  std::unordered_map<String, ObjectRef> key2default_;
  /*! \brief Index used for internal lookup of attribute registry */
  uint32_t index_;
  friend void TargetValidateSchema(const Map<String, ObjectRef>&);
  friend class Target;
  friend class TargetId;
  template <typename, typename>
  friend class AttrRegistry;
  template <typename>
  friend class AttrRegistryMapContainerMap;
  friend class TargetIdRegEntry;
  template <typename, typename, typename>
  friend struct detail::ValueTypeInfoMaker;
};

/*!
 * \brief Managed reference class to TargetIdNode
 * \sa TargetIdNode
 */
class TargetId : public ObjectRef {
 public:
  TargetId() = default;
  /*! \brief Get the attribute map given the attribute name */
  template <typename ValueType>
  static inline TargetIdAttrMap<ValueType> GetAttrMap(const String& attr_name);
  /*!
   * \brief Retrieve the TargetId given its name
   * \param target_id_name Name of the target id
   * \return The TargetId requested
   */
  TVM_DLL static const TargetId& Get(const String& target_id_name);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TargetId, ObjectRef, TargetIdNode);

 private:
  /*! \brief Mutable access to the container class  */
  TargetIdNode* operator->() { return static_cast<TargetIdNode*>(data_.get()); }
  TVM_DLL static const AttrRegistryMapContainerMap<TargetId>& GetAttrMapContainer(
      const String& attr_name);
  template <typename, typename>
  friend class AttrRegistry;
  friend class TargetIdRegEntry;
  friend class Target;
};

/*!
 * \brief Map<TargetId, ValueType> used to store meta-information about TargetId
 * \tparam ValueType The type of the value stored in map
 */
template <typename ValueType>
class TargetIdAttrMap : public AttrRegistryMap<TargetId, ValueType> {
 public:
  using TParent = AttrRegistryMap<TargetId, ValueType>;
  using TParent::count;
  using TParent::get;
  using TParent::operator[];
  explicit TargetIdAttrMap(const AttrRegistryMapContainerMap<TargetId>& map) : TParent(map) {}
};

/*!
 * \brief Helper structure to register TargetId
 * \sa TVM_REGISTER_TARGET_ID
 */
class TargetIdRegEntry {
 public:
  /*!
   * \brief Register additional attributes to target_id.
   * \param attr_name The name of the attribute.
   * \param value The value to be set.
   * \param plevel The priority level of this attribute,
   *  an higher priority level attribute
   *  will replace lower priority level attribute.
   *  Must be bigger than 0.
   *
   *  Cannot set with same plevel twice in the code.
   *
   * \tparam ValueType The type of the value to be set.
   */
  template <typename ValueType>
  inline TargetIdRegEntry& set_attr(const String& attr_name, const ValueType& value,
                                    int plevel = 10);
  /*!
   * \brief Set DLPack's device_type the target
   * \param device_type Device type
   */
  inline TargetIdRegEntry& set_device_type(int device_type);
  /*!
   * \brief Set DLPack's device_type the target
   * \param keys The default keys
   */
  inline TargetIdRegEntry& set_default_keys(std::vector<String> keys);
  /*!
   * \brief Register a valid configuration option and its ValueType for validation
   * \param key The configuration key
   * \tparam ValueType The value type to be registered
   */
  template <typename ValueType>
  inline TargetIdRegEntry& add_attr_option(const String& key);
  /*!
   * \brief Register a valid configuration option and its ValueType for validation
   * \param key The configuration key
   * \param default_value The default value of the key
   * \tparam ValueType The value type to be registered
   */
  template <typename ValueType>
  inline TargetIdRegEntry& add_attr_option(const String& key, ObjectRef default_value);
  /*! \brief Set name of the TargetId to be the same as registry if it is empty */
  inline TargetIdRegEntry& set_name();
  /*!
   * \brief Register or get a new entry.
   * \param target_id_name The name of the TargetId.
   * \return the corresponding entry.
   */
  TVM_DLL static TargetIdRegEntry& RegisterOrGet(const String& target_id_name);

 private:
  TargetId id_;
  String name;

  /*! \brief private constructor */
  explicit TargetIdRegEntry(uint32_t reg_index) : id_(make_object<TargetIdNode>()) {
    id_->index_ = reg_index;
  }
  /*!
   * \brief update the attribute TargetIdAttrMap
   * \param key The name of the attribute
   * \param value The value to be set
   * \param plevel The priority level
   */
  TVM_DLL void UpdateAttr(const String& key, TVMRetValue value, int plevel);
  template <typename, typename>
  friend class AttrRegistry;
  friend class TargetId;
};

#define TVM_TARGET_ID_REGISTER_VAR_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::TargetIdRegEntry& __make_##TargetId

/*!
 * \def TVM_REGISTER_TARGET_ID
 * \brief Register a new target id, or set attribute of the corresponding target id.
 *
 * \param TargetIdName The name of target id
 *
 * \code
 *
 *  TVM_REGISTER_TARGET_ID("llvm")
 *  .set_attr<TPreCodegenPass>("TPreCodegenPass", a-pre-codegen-pass)
 *  .add_attr_option<Bool>("system_lib")
 *  .add_attr_option<String>("mtriple")
 *  .add_attr_option<String>("mattr");
 *
 * \endcode
 */
#define TVM_REGISTER_TARGET_ID(TargetIdName)                    \
  TVM_STR_CONCAT(TVM_TARGET_ID_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::TargetIdRegEntry::RegisterOrGet(TargetIdName).set_name()

namespace detail {
template <typename Type, template <typename...> class Container>
struct is_specialized : std::false_type {
  using type = std::false_type;
};

template <template <typename...> class Container, typename... Args>
struct is_specialized<Container<Args...>, Container> : std::true_type {
  using type = std::true_type;
};

template <typename ValueType, typename IsArray = typename is_specialized<ValueType, Array>::type,
          typename IsMap = typename is_specialized<ValueType, Map>::type>
struct ValueTypeInfoMaker {};

template <typename ValueType>
struct ValueTypeInfoMaker<ValueType, std::false_type, std::false_type> {
  using ValueTypeInfo = TargetIdNode::ValueTypeInfo;

  ValueTypeInfo operator()() const {
    uint32_t tindex = ValueType::ContainerType::_GetOrAllocRuntimeTypeIndex();
    ValueTypeInfo info;
    info.type_index = tindex;
    info.type_key = runtime::Object::TypeIndex2Key(tindex);
    info.key = nullptr;
    info.val = nullptr;
    return info;
  }
};

template <typename ValueType>
struct ValueTypeInfoMaker<ValueType, std::true_type, std::false_type> {
  using ValueTypeInfo = TargetIdNode::ValueTypeInfo;

  ValueTypeInfo operator()() const {
    using key_type = ValueTypeInfoMaker<typename ValueType::value_type>;
    uint32_t tindex = ValueType::ContainerType::_GetOrAllocRuntimeTypeIndex();
    ValueTypeInfo info;
    info.type_index = tindex;
    info.type_key = runtime::Object::TypeIndex2Key(tindex);
    info.key = std::unique_ptr<ValueTypeInfo>(new ValueTypeInfo(key_type()()));
    info.val = nullptr;
    return info;
  }
};

template <typename ValueType>
struct ValueTypeInfoMaker<ValueType, std::false_type, std::true_type> {
  using ValueTypeInfo = TargetIdNode::ValueTypeInfo;
  ValueTypeInfo operator()() const {
    using key_type = ValueTypeInfoMaker<typename ValueType::key_type>;
    using val_type = ValueTypeInfoMaker<typename ValueType::mapped_type>;
    uint32_t tindex = ValueType::ContainerType::_GetOrAllocRuntimeTypeIndex();
    ValueTypeInfo info;
    info.type_index = tindex;
    info.type_key = runtime::Object::TypeIndex2Key(tindex);
    info.key = std::unique_ptr<ValueTypeInfo>(new ValueTypeInfo(key_type()()));
    info.val = std::unique_ptr<ValueTypeInfo>(new ValueTypeInfo(val_type()()));
    return info;
  }
};

}  // namespace detail

template <typename ValueType>
inline TargetIdAttrMap<ValueType> TargetId::GetAttrMap(const String& attr_name) {
  return TargetIdAttrMap<ValueType>(GetAttrMapContainer(attr_name));
}

template <typename ValueType>
inline TargetIdRegEntry& TargetIdRegEntry::set_attr(const String& attr_name, const ValueType& value,
                                                    int plevel) {
  CHECK_GT(plevel, 0) << "plevel in set_attr must be greater than 0";
  runtime::TVMRetValue rv;
  rv = value;
  UpdateAttr(attr_name, rv, plevel);
  return *this;
}

inline TargetIdRegEntry& TargetIdRegEntry::set_device_type(int device_type) {
  id_->device_type = device_type;
  return *this;
}

inline TargetIdRegEntry& TargetIdRegEntry::set_default_keys(std::vector<String> keys) {
  id_->default_keys = keys;
  return *this;
}

template <typename ValueType>
inline TargetIdRegEntry& TargetIdRegEntry::add_attr_option(const String& key) {
  CHECK(!id_->key2vtype_.count(key))
      << "AttributeError: add_attr_option failed because '" << key << "' has been set once";
  id_->key2vtype_[key] = detail::ValueTypeInfoMaker<ValueType>()();
  return *this;
}

template <typename ValueType>
inline TargetIdRegEntry& TargetIdRegEntry::add_attr_option(const String& key,
                                                           ObjectRef default_value) {
  add_attr_option<ValueType>(key);
  id_->key2default_[key] = default_value;
  return *this;
}

inline TargetIdRegEntry& TargetIdRegEntry::set_name() {
  if (id_->name.empty()) {
    id_->name = name;
  }
  return *this;
}

}  // namespace tvm

#endif  // TVM_TARGET_TARGET_ID_H_
