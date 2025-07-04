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
 * \file tvm/ir/attrs.h
 * \brief Helpers for attribute objects.
 *
 *  This module enables declaration of named attributes
 *  which support default value setup and bound checking.
 *
 * \sa AttrsNode, TVM_DECLARE_ATTRS, TVM_ATTR_FIELD
 */
#ifndef TVM_IR_ATTRS_H_
#define TVM_IR_ATTRS_H_

#include <dmlc/common.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/reflection.h>
#include <tvm/ir/expr.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>

#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {

/*!
 * \brief Create a NodeRef type that represents null.
 * \tparam TNodeRef the type to be created.
 * \return A instance that will represent None.
 */
template <typename TObjectRef>
inline TObjectRef NullValue() {
  static_assert(TObjectRef::_type_is_nullable, "Can only get NullValue for nullable types");
  return TObjectRef(ObjectPtr<Object>(nullptr));
}

template <>
inline DataType NullValue<DataType>() {
  return DataType(DataType::kHandle, 0, 0);
}

/*!
 * \brief Information about attribute fields in string representations.
 */
class AttrFieldInfoNode : public Object {
 public:
  /*! \brief name of the field */
  String name;
  /*! \brief type docstring information in str. */
  String type_info;
  /*! \brief detailed description of the type */
  String description;

  static void RegisterReflection() {
    namespace rfl = ffi::reflection;
    rfl::ObjectDef<AttrFieldInfoNode>()
        .def_ro("name", &AttrFieldInfoNode::name)
        .def_ro("type_info", &AttrFieldInfoNode::type_info)
        .def_ro("description", &AttrFieldInfoNode::description);
  }

  static constexpr const char* _type_key = "ir.AttrFieldInfo";

  static constexpr bool _type_has_method_sequal_reduce = false;
  static constexpr bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrFieldInfoNode, Object);
};

/*! \brief AttrFieldInfo */
class AttrFieldInfo : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(AttrFieldInfo, ObjectRef, AttrFieldInfoNode);
};

/*!
 * \brief Base class of all attribute class
 * \note Do not subclass AttrBaseNode directly,
 *       subclass AttrsNode instead.
 * \sa AttrsNode
 */
class BaseAttrsNode : public Object {
 public:
  /*! \brief virtual destructor */
  virtual ~BaseAttrsNode() {}
  /*!
   * \brief Initialize the attributes by sequence of arguments
   * \param args The positional arguments in the form
   *        [key0, value0, key1, value1, ..., key_n, value_n]
   */
  template <typename... Args>
  inline void InitBySeq(Args&&... args);
  /*!
   * \brief Initialize the attributes by arguments.
   * \param kwargs The key value pairs for initialization.
   *        [key0, value0, key1, value1, ..., key_n, value_n]
   * \param allow_unknown Whether allow additional unknown fields.
   * \note This function throws when the required field is not present.
   */
  TVM_DLL virtual void InitByPackedArgs(const ffi::PackedArgs& kwargs,
                                        bool allow_unknown = false) = 0;

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const char* _type_key = "ir.Attrs";
  TVM_DECLARE_BASE_OBJECT_INFO(BaseAttrsNode, Object);
};

/*!
 * \brief Managed reference to BaseAttrsNode.
 * \sa AttrsNode, BaseAttrsNode
 */
class Attrs : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Attrs, ObjectRef, BaseAttrsNode);
};

/*!
 * \brief Specialized attribute type that is backed by a map.
 *  The DictAttrsNode implements the Attrs behavior,
 *  its fields are directly accessible via object.field_name
 *  like other normal nodes.
 */
class DictAttrsNode : public BaseAttrsNode {
 public:
  /*! \brief internal attrs map */
  Map<String, ffi::Any> dict;

  static void RegisterReflection() {
    namespace rfl = ffi::reflection;
    rfl::ObjectDef<DictAttrsNode>().def_ro("__dict__", &DictAttrsNode::dict);
  }

  bool SEqualReduce(const DictAttrsNode* other, SEqualReducer equal) const {
    return equal(dict, other->dict);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(dict); }
  void InitByPackedArgs(const ffi::PackedArgs& args, bool allow_unknown) final;

  // type info
  static constexpr const char* _type_key = "ir.DictAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(DictAttrsNode, BaseAttrsNode);
};

/*!
 * \brief Managed reference to DictAttrsNode
 * \sa DictAttrsNode.
 */
class DictAttrs : public Attrs {
 public:
  /*!
   * \brief Consruct a Attrs backed by DictAttrsNode.
   * \param dict The attributes.
   */
  TVM_DLL explicit DictAttrs(Map<String, Any> dict = {});

  // Utils for accessing attributes
  // This needs to be on DictAttrs, not DictAttrsNode because we return the default
  // value if DictAttrsNode is not defined.
  /*!
   * \brief Get a function attribute.
   *
   * \param attr_key The attribute key.
   * \param default_value The default value if the key does not exist, defaults to nullptr.
   *
   * \return The result
   *
   * \tparam TOBjectRef the expected object type.
   * \throw Error if the key exists but the value does not match TObjectRef
   *
   * \code
   *
   *  void GetAttrExample(const BaseFunc& f) {
   *    auto value = f->attrs.GetAttr<Integer>("AttrKey", 0);
   *  }
   *
   * \endcode
   */
  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(
      const std::string& attr_key,
      Optional<TObjectRef> default_value = Optional<TObjectRef>(std::nullopt)) const {
    if (!defined()) return default_value;
    const DictAttrsNode* node = this->as<DictAttrsNode>();
    auto it = node->dict.find(attr_key);
    if (it != node->dict.end()) {
      return (*it).second.cast<TObjectRef>();
    } else {
      return default_value;
    }
  }
  // variant that uses TObjectRef to enable implicit conversion to default value.
  template <typename TObjectRef>
  Optional<TObjectRef> GetAttr(const std::string& attr_key, TObjectRef default_value) const {
    return GetAttr<TObjectRef>(attr_key, Optional<TObjectRef>(default_value));
  }
  /*!
   * \brief Check whether the function has an non-zero integer attr.
   *
   * This function can be used to check whether an optional
   * attribute mark(e.g. inline) exists.
   *
   * \param attr_key The key to the attribute.
   * \return The check result.
   *
   * \code
   *
   *  void HasNonzeroAttrExample(const BaseFunc& f) {
   *    if (f->HasNonzeroAttr(attr::kInline)) {
   *      // inline the function.
   *    }
   *  }
   *
   * \endcode
   */
  bool HasNonzeroAttr(const std::string& attr_key) const {
    return GetAttr<Integer>(attr_key, 0).value_or(0).IntValue() != 0;
  }

  TVM_DEFINE_OBJECT_REF_METHODS_WITHOUT_DEFAULT_CONSTRUCTOR(DictAttrs, Attrs, DictAttrsNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DictAttrsNode);
};

/*!
 * \brief Copy the DictAttrs, but overrides attributes with the
 * entries from \p attrs.
 *
 * \param attrs The DictAttrs to update
 *
 * \param new_attrs Key/values attributes to add to \p attrs.
 *
 * \returns The new DictAttrs with updated attributes.
 */
DictAttrs WithAttrs(DictAttrs attrs, Map<String, Any> new_attrs);

/*!
 * \brief Copy the DictAttrs, but overrides a single attribute.
 *
 * \param attrs The DictAttrs to update
 *
 * \param key The update to insert or update.
 *
 * \param value The new value of the attribute
 *
 * \returns The new DictAttrs with updated attributes.
 */
DictAttrs WithAttr(DictAttrs attrs, String key, Any value);

inline DictAttrs WithAttr(DictAttrs attrs, const std::string& key, Any value) {
  return WithAttr(std::move(attrs), String(key), std::move(value));
}

/*!
 * \brief Copy the DictAttrs, but without a specific attribute.
 *
 * \param attrs The DictAttrs to update
 *
 * \param key The key to remove
 *
 * \returns The new DictAttrs with updated attributes.
 */
DictAttrs WithoutAttr(DictAttrs attrs, const std::string& key);

/*!
 * \brief Copy the function or module, but overrides
 *        the attribute value key with the value.
 *
 * \param input The thing to annotate (BaseFunc or IRModule)
 * \param attr_key The attribute key.
 * \param attr_value The value attribute value.
 *
 * \tparam TFunc The corresponding function or module type.
 *
 * \returns The new function or module with updated attributes.
 *
 * \note This function performs copy on write optimization for func and module.
 *       If we move a uniquely referenced func or module into WithAttr,
 *       then no additional copy will be performed.
 *
 *       This is also why we make it as a function instead of a member function
 *       and why we pass by value in the first argument.
 *
 * \code
 *
 *  // Recommended way to trigger copy on write
 *  func = WithAttr(std::move(func), "key1", value1);
 *  func = WithAttr(std::move(func), "key2", value2);
 *
 * \endcode
 */
template <typename TFunc>
inline TFunc WithAttr(TFunc input, const std::string& attr_key, Any attr_value) {
  using TNode = typename TFunc::ContainerType;
  static_assert(TNode::_type_final, "Can only operate on the leaf nodes");
  TNode* node = input.CopyOnWrite();
  node->attrs = WithAttr(std::move(node->attrs), attr_key, attr_value);
  return input;
}

/*!
 * \brief Copy the function or module, but overrides the attributes with the entries from \p attrs.
 *
 * \param input The thing to annotate (BaseFunc or IRModule)
 * \param attrs Key/values attributes to add to \p input.
 *
 * \tparam TFunc The corresponding function or module type.
 *
 * \returns The new function or module with updated attributes.
 */
template <typename TFunc>
inline TFunc WithAttrs(TFunc input, Map<String, Any> attrs) {
  using TNode = typename TFunc::ContainerType;
  static_assert(TNode::_type_final, "Can only operate on the leaf nodes");
  TNode* node = input.CopyOnWrite();

  node->attrs = WithAttrs(std::move(node->attrs), attrs);

  return input;
}

/*!
 * \brief Copy the function or module, but removes the specified
 *        attribute.
 *
 * \param input The thing to annotate (BaseFunc or IRModule)
 * \param attr_key The attribute key.
 *
 * \tparam TFunc The corresponding function or module type.
 *
 * \returns The new function or module with removed attribute.
 *
 * \note This function performs copy on write optimization for func and module.
 *       If we move a uniquely referenced func or module into WithoutAttr,
 *       then no additional copy will be performed.
 *
 *       This is also why we make it as a function instead of a member function
 *       and why we pass by value in the first argument.
 *
 * \code
 *
 *  // Recommended way to trigger copy on write
 *  func = WithoutAttr(std::move(func), "key1");
 *  func = WithoutAttr(std::move(func), "key2");
 *
 * \endcode
 */
template <typename TFunc>
inline TFunc WithoutAttr(TFunc input, const std::string& attr_key) {
  using TNode = typename TFunc::ContainerType;
  static_assert(TNode::_type_final, "Can only operate on the leaf nodes");

  TNode* node = input.CopyOnWrite();
  node->attrs = WithoutAttr(std::move(node->attrs), attr_key);

  return input;
}

/*!
 * \brief Adapter for AttrsNode with the new reflection API.
 *
 * We will phaseout the old AttrsNode in future in favor of the new reflection API.
 * This adapter allows us to gradually migrate to the new reflection API.
 *
 * \tparam DerivedType The final attribute type.
 */
template <typename DerivedType>
class AttrsNodeReflAdapter : public BaseAttrsNode {
 public:
  void InitByPackedArgs(const ffi::PackedArgs& args, bool allow_unknown) final {
    LOG(FATAL) << "`" << DerivedType::_type_key << "` uses new reflection mechanism for init";
  }

  bool SEqualReduce(const DerivedType* other, SEqualReducer equal) const {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(DerivedType::RuntimeTypeIndex());
    bool success = true;
    ffi::reflection::ForEachFieldInfoWithEarlyStop(
        type_info, [&](const TVMFFIFieldInfo* field_info) {
          ffi::reflection::FieldGetter field_getter(field_info);
          ffi::Any field_value = field_getter(self());
          ffi::Any other_field_value = field_getter(other);
          if (!equal.AnyEqual(field_value, other_field_value)) {
            success = false;
            return true;
          }
          return false;
        });
    return success;
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(DerivedType::RuntimeTypeIndex());
    ffi::reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
      ffi::reflection::FieldGetter field_getter(field_info);
      ffi::Any field_value = field_getter(self());
      hash_reducer(field_value);
    });
  }

 private:
  DerivedType* self() const {
    return const_cast<DerivedType*>(static_cast<const DerivedType*>(this));
  }
};

/*!
 * \brief Create an Attr object with all default values.
 * \tparam TAttrNode the type to be created.
 * \return A instance that will represent None.
 */
template <typename TAttrs>
inline TAttrs AttrsWithDefaultValues() {
  static_assert(std::is_base_of_v<Attrs, TAttrs>, "Can only take attr nodes");
  using ContainerType = typename TAttrs::ContainerType;
  if constexpr (std::is_base_of_v<AttrsNodeReflAdapter<ContainerType>, ContainerType>) {
    static auto finit_object = ffi::Function::GetGlobalRequired("ffi.MakeObjectFromPackedArgs");
    AnyView packed_args[1];
    packed_args[0] = ContainerType::RuntimeTypeIndex();
    ffi::Any rv;
    finit_object.CallPacked(ffi::PackedArgs(packed_args, 1), &rv);
    return rv.cast<TAttrs>();
  } else {
    auto n = make_object<ContainerType>();
    n->InitByPackedArgs(ffi::PackedArgs(nullptr, 0), false);
    return TAttrs(n);
  }
}

}  // namespace tvm
#endif  // TVM_IR_ATTRS_H_
