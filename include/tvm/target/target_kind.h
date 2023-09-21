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

#include <tvm/node/attr_registry_map.h>
#include <tvm/node/node.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {

class Target;

/*!
 * \brief Map containing parsed features of a specific Target
 */
using TargetFeatures = Map<String, ObjectRef>;

/*!
 * \brief TargetParser to apply on instantiation of a given TargetKind
 *
 * \param target_json Target in JSON format to be transformed during parsing.
 *
 * \return The transformed Target JSON object.
 */
using TargetJSON = Map<String, ObjectRef>;
using FTVMTargetParser = runtime::TypedPackedFunc<TargetJSON(TargetJSON)>;

namespace detail {
template <typename, typename, typename>
struct ValueTypeInfoMaker;
}

class TargetInternal;

template <typename>
class TargetKindAttrMap;

/*! \brief Target kind, specifies the kind of the target */
class TargetKindNode : public Object {
 public:
  /*! \brief Name of the target kind */
  String name;
  /*! \brief Device type of target kind */
  int default_device_type;
  /*! \brief Default keys of the target */
  Array<String> default_keys;
  /*! \brief Function used to preprocess on target creation */
  PackedFunc preprocessor;
  /*! \brief Function used to parse a JSON target during creation */
  FTVMTargetParser target_parser;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("default_device_type", &default_device_type);
    v->Visit("default_keys", &default_keys);
  }

  static constexpr const char* _type_key = "TargetKind";
  TVM_DECLARE_FINAL_OBJECT_INFO(TargetKindNode, Object);

 private:
  /*! \brief Return the index stored in attr registry */
  uint32_t AttrRegistryIndex() const { return index_; }
  /*! \brief Return the name stored in attr registry */
  String AttrRegistryName() const { return name; }
  /*! \brief Stores the required type_key and type_index of a specific attr of a target */
  struct ValueTypeInfo {
    String type_key;
    uint32_t type_index;
    std::unique_ptr<ValueTypeInfo> key;
    std::unique_ptr<ValueTypeInfo> val;
  };
  /*! \brief A hash table that stores the type information of each attr of the target key */
  std::unordered_map<String, ValueTypeInfo> key2vtype_;
  /*! \brief A hash table that stores the default value of each attr of the target key */
  std::unordered_map<String, ObjectRef> key2default_;
  /*! \brief Index used for internal lookup of attribute registry */
  uint32_t index_;

  template <typename, typename, typename>
  friend struct detail::ValueTypeInfoMaker;
  template <typename, typename>
  friend class AttrRegistry;
  template <typename>
  friend class AttrRegistryMapContainerMap;
  friend class TargetKindRegEntry;
  friend class TargetInternal;
};

/*!
 * \brief Managed reference class to TargetKindNode
 * \sa TargetKindNode
 */
class TargetKind : public ObjectRef {
 public:
  TargetKind() = default;
  /*! \brief Get the attribute map given the attribute name */
  template <typename ValueType>
  static inline TargetKindAttrMap<ValueType> GetAttrMap(const String& attr_name);
  /*!
   * \brief Retrieve the TargetKind given its name
   * \param target_kind_name Name of the target kind
   * \return The TargetKind requested
   */
  TVM_DLL static Optional<TargetKind> Get(const String& target_kind_name);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TargetKind, ObjectRef, TargetKindNode);
  /*! \brief Mutable access to the container class  */
  TargetKindNode* operator->() { return static_cast<TargetKindNode*>(data_.get()); }

 private:
  TVM_DLL static const AttrRegistryMapContainerMap<TargetKind>& GetAttrMapContainer(
      const String& attr_name);
  friend class TargetKindRegEntry;
  friend class TargetInternal;
};

/*!
 * \brief Map<TargetKind, ValueType> used to store meta-information about TargetKind
 * \tparam ValueType The type of the value stored in map
 */
template <typename ValueType>
class TargetKindAttrMap : public AttrRegistryMap<TargetKind, ValueType> {
 public:
  using TParent = AttrRegistryMap<TargetKind, ValueType>;
  using TParent::count;
  using TParent::get;
  using TParent::operator[];
  explicit TargetKindAttrMap(const AttrRegistryMapContainerMap<TargetKind>& map) : TParent(map) {}
};

/*! \brief Value used with --runtime in target specs to indicate the C++ runtime. */
static constexpr const char* kTvmRuntimeCpp = "c++";

/*! \brief Value used with --runtime in target specs to indicate the C runtime. */
static constexpr const char* kTvmRuntimeCrt = "c";

/*!
 * \brief Helper structure to register TargetKind
 * \sa TVM_REGISTER_TARGET_KIND
 */
class TargetKindRegEntry {
 public:
  /*!
   * \brief Register additional attributes to target_kind.
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
  inline TargetKindRegEntry& set_attr(const String& attr_name, const ValueType& value,
                                      int plevel = 10);
  /*!
   * \brief Set DLPack's device_type the target
   * \param device_type Device type
   */
  inline TargetKindRegEntry& set_default_device_type(int device_type);
  /*!
   * \brief Set DLPack's device_type the target
   * \param keys The default keys
   */
  inline TargetKindRegEntry& set_default_keys(std::vector<String> keys);
  /*!
   * \brief Set the pre-processing function applied upon target creation
   * \tparam FLambda Type of the function
   * \param f The pre-processing function
   */
  template <typename FLambda>
  inline TargetKindRegEntry& set_attrs_preprocessor(FLambda f);
  /*!
   * \brief Set the parsing function applied upon target creation
   * \param parser The Target parsing function
   */
  inline TargetKindRegEntry& set_target_parser(FTVMTargetParser parser);
  /*!
   * \brief Register a valid configuration option and its ValueType for validation
   * \param key The configuration key
   * \tparam ValueType The value type to be registered
   */
  template <typename ValueType>
  inline TargetKindRegEntry& add_attr_option(const String& key);
  /*!
   * \brief Register a valid configuration option and its ValueType for validation
   * \param key The configuration key
   * \param default_value The default value of the key
   * \tparam ValueType The value type to be registered
   */
  template <typename ValueType>
  inline TargetKindRegEntry& add_attr_option(const String& key, ObjectRef default_value);
  /*! \brief Set name of the TargetKind to be the same as registry if it is empty */
  inline TargetKindRegEntry& set_name();
  /*!
   * \brief List all the entry names in the registry.
   * \return The entry names.
   */
  TVM_DLL static Array<String> ListTargetKinds();
  /*!
   * \brief Get all supported option names and types for a given Target kind.
   * \return Map of option name to type
   */
  TVM_DLL static Map<String, String> ListTargetKindOptions(const TargetKind& kind);

  /*!
   * \brief Register or get a new entry.
   * \param target_kind_name The name of the TargetKind.
   * \return the corresponding entry.
   */
  TVM_DLL static TargetKindRegEntry& RegisterOrGet(const String& target_kind_name);

 private:
  TargetKind kind_;
  String name;

  /*! \brief private constructor */
  explicit TargetKindRegEntry(uint32_t reg_index) : kind_(make_object<TargetKindNode>()) {
    kind_->index_ = reg_index;
  }
  /*!
   * \brief update the attribute TargetKindAttrMap
   * \param key The name of the attribute
   * \param value The value to be set
   * \param plevel The priority level
   */
  TVM_DLL void UpdateAttr(const String& key, TVMRetValue value, int plevel);
  template <typename, typename>
  friend class AttrRegistry;
  friend class TargetKind;
};

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
  using ValueTypeInfo = TargetKindNode::ValueTypeInfo;

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
  using ValueTypeInfo = TargetKindNode::ValueTypeInfo;

  ValueTypeInfo operator()() const {
    using key_type = ValueTypeInfoMaker<typename ValueType::value_type>;
    uint32_t tindex = ValueType::ContainerType::_GetOrAllocRuntimeTypeIndex();
    ValueTypeInfo info;
    info.type_index = tindex;
    info.type_key = runtime::Object::TypeIndex2Key(tindex);
    info.key = std::make_unique<ValueTypeInfo>(key_type()());
    info.val = nullptr;
    return info;
  }
};

template <typename ValueType>
struct ValueTypeInfoMaker<ValueType, std::false_type, std::true_type> {
  using ValueTypeInfo = TargetKindNode::ValueTypeInfo;
  ValueTypeInfo operator()() const {
    using key_type = ValueTypeInfoMaker<typename ValueType::key_type>;
    using val_type = ValueTypeInfoMaker<typename ValueType::mapped_type>;
    uint32_t tindex = ValueType::ContainerType::_GetOrAllocRuntimeTypeIndex();
    ValueTypeInfo info;
    info.type_index = tindex;
    info.type_key = runtime::Object::TypeIndex2Key(tindex);
    info.key = std::make_unique<ValueTypeInfo>(key_type()());
    info.val = std::make_unique<ValueTypeInfo>(val_type()());
    return info;
  }
};

}  // namespace detail

template <typename ValueType>
inline TargetKindAttrMap<ValueType> TargetKind::GetAttrMap(const String& attr_name) {
  return TargetKindAttrMap<ValueType>(GetAttrMapContainer(attr_name));
}

template <typename ValueType>
inline TargetKindRegEntry& TargetKindRegEntry::set_attr(const String& attr_name,
                                                        const ValueType& value, int plevel) {
  ICHECK_GT(plevel, 0) << "plevel in set_attr must be greater than 0";
  runtime::TVMRetValue rv;
  rv = value;
  UpdateAttr(attr_name, rv, plevel);
  return *this;
}

inline TargetKindRegEntry& TargetKindRegEntry::set_default_device_type(int device_type) {
  kind_->default_device_type = device_type;
  return *this;
}

inline TargetKindRegEntry& TargetKindRegEntry::set_default_keys(std::vector<String> keys) {
  kind_->default_keys = keys;
  return *this;
}

template <typename FLambda>
inline TargetKindRegEntry& TargetKindRegEntry::set_attrs_preprocessor(FLambda f) {
  LOG(WARNING) << "set_attrs_preprocessor is deprecated please use set_target_parser instead";
  using FType = typename tvm::runtime::detail::function_signature<FLambda>::FType;
  kind_->preprocessor = tvm::runtime::TypedPackedFunc<FType>(std::move(f)).packed();
  return *this;
}

inline TargetKindRegEntry& TargetKindRegEntry::set_target_parser(FTVMTargetParser parser) {
  kind_->target_parser = parser;
  return *this;
}

template <typename ValueType>
inline TargetKindRegEntry& TargetKindRegEntry::add_attr_option(const String& key) {
  ICHECK(!kind_->key2vtype_.count(key))
      << "AttributeError: add_attr_option failed because '" << key << "' has been set once";
  kind_->key2vtype_[key] = detail::ValueTypeInfoMaker<ValueType>()();
  return *this;
}

template <typename ValueType>
inline TargetKindRegEntry& TargetKindRegEntry::add_attr_option(const String& key,
                                                               ObjectRef default_value) {
  add_attr_option<ValueType>(key);
  kind_->key2default_[key] = default_value;
  return *this;
}

inline TargetKindRegEntry& TargetKindRegEntry::set_name() {
  if (kind_->name.empty()) {
    kind_->name = name;
  }
  return *this;
}

#define TVM_TARGET_KIND_REGISTER_VAR_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::TargetKindRegEntry& __make_##TargetKind

namespace attr {
//
// Distinguished TargetKind attribute names.
//

/*!
 * \brief A \p TargetKind attribute of type \p Bool. If true, then the target kind name also
 * corresponds to an external codegen 'compiler' name. That name may be used:
 *  - To retrieve partitioning rules using \p get_partition_table.
 *  - To attach to Relay Functions under the \p attr::kCompiler attribute to indicate
 *    the function is to be compiled by the external codegen path.
 *
 * The \p CollagePartition pass uses this attribute to guide it's search over candidate partitions
 * using external codegen.
 *
 * See also \p Target::IsExternalCodegenFor
 */
constexpr const char* kIsExternalCodegen = "is_external_codegen";

/*!
 * \brief A \p TargetKind attribute of type \p FTVMRelayToTIR. If set, then the target kind name
 * also corresponds to an external codegen 'compiler' name, and the bound value is a \p Pass
 * to apply before the TVM lowering.
 *
 * See also \p Target::IsExternalCodegenFor
 */
constexpr const char* kRelayToTIR = "RelayToTIR";

}  // namespace attr

/*!
 * \def TVM_REGISTER_TARGET_KIND
 * \brief Register a new target kind, or set attribute of the corresponding target kind.
 *
 * \param TargetKindName The name of target kind
 * \param DeviceType The DLDeviceType of the target kind
 *
 * \code
 *
 *  TVM_REGISTER_TARGET_KIND("llvm")
 *  .set_attr<TPreCodegenPass>("TPreCodegenPass", a-pre-codegen-pass)
 *  .add_attr_option<Bool>("system_lib")
 *  .add_attr_option<String>("mtriple")
 *  .add_attr_option<String>("mattr");
 *
 * \endcode
 */
#define TVM_REGISTER_TARGET_KIND(TargetKindName, DeviceType)      \
  TVM_STR_CONCAT(TVM_TARGET_KIND_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::TargetKindRegEntry::RegisterOrGet(TargetKindName)    \
          .set_name()                                             \
          .set_default_device_type(DeviceType)                    \
          .add_attr_option<Array<String>>("keys")                 \
          .add_attr_option<String>("tag")                         \
          .add_attr_option<String>("device")                      \
          .add_attr_option<String>("model")                       \
          .add_attr_option<Array<String>>("libs")                 \
          .add_attr_option<Target>("host")                        \
          .add_attr_option<Integer>("from_device")                \
          .add_attr_option<Integer>("target_device_type")

}  // namespace tvm

#endif  // TVM_TARGET_TARGET_KIND_H_
