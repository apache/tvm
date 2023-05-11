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
 * \code
 *   struct MyAttrs : public tvm::AttrsNode<MyAttrs> {
 *     float learning_rate;
 *     int num_hidden;
 *     String name;
 *     // declare attribute fields in header file
 *     TVM_DECLARE_ATTRS(MyAttrs, "attrs.MyAttrs") {
 *       TVM_ATTR_FIELD(num_hidden).set_lower_bound(1);
 *       TVM_ATTR_FIELD(learning_rate).set_default(0.01f);
 *       TVM_ATTR_FIELD(name).set_default("hello");
 *     }
 *   };
 *   // register it in cc file
 *   TVM_REGISTER_NODE_TYPE(MyAttrs);
 * \endcode
 *
 * \sa AttrsNode, TVM_DECLARE_ATTRS, TVM_ATTR_FIELD
 */
#ifndef TVM_IR_ATTRS_H_
#define TVM_IR_ATTRS_H_

#include <dmlc/common.h>
#include <tvm/ir/expr.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/packed_func.h>

#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
/*!
 * \brief Declare an attribute function.
 * \param ClassName The name of the class.
 * \param TypeKey The type key to be used by the TVM node system.
 */
#define TVM_DECLARE_ATTRS(ClassName, TypeKey)                    \
  static constexpr const char* _type_key = TypeKey;              \
  TVM_DECLARE_FINAL_OBJECT_INFO(ClassName, ::tvm::BaseAttrsNode) \
  template <typename FVisit>                                     \
  void _tvm_VisitAttrs(FVisit& _tvm_fvisit)  // NOLINT(*)

/*!
 * \brief Declare an attribute field.
 * \param FieldName The field name.
 */
#define TVM_ATTR_FIELD(FieldName) _tvm_fvisit(#FieldName, &FieldName)

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

/*! \brief Error thrown during attribute checking. */
struct AttrError : public Error {
  /*!
   * \brief constructor
   * \param msg error message
   */
  explicit AttrError(std::string msg) : Error("AttributeError:" + msg) {}
};

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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("type_info", &type_info);
    v->Visit("description", &description);
  }

  static constexpr const char* _type_key = "AttrFieldInfo";
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
  using TVMArgs = runtime::TVMArgs;
  using TVMRetValue = runtime::TVMRetValue;
  /*! \brief virtual destructor */
  virtual ~BaseAttrsNode() {}
  // visit function
  virtual void VisitAttrs(AttrVisitor* v) {}
  /*!
   * \brief Initialize the attributes by sequence of arguments
   * \param args The positional arguments in the form
   *        [key0, value0, key1, value1, ..., key_n, value_n]
   */
  template <typename... Args>
  inline void InitBySeq(Args&&... args);
  /*!
   * \brief Print readible docstring to ostream, add newline.
   * \param os the stream to print the docstring to.
   */
  inline void PrintDocString(std::ostream& os) const;  // NOLINT(*)
  /*!
   * \brief Visit attributes that do not equal the default value.
   *
   * \note This is useful to extract fields for concise printing.
   * \param v The visitor
   */
  TVM_DLL virtual void VisitNonDefaultAttrs(AttrVisitor* v) = 0;
  /*!
   * \brief Get the field information
   * \return The fields in the Attrs.
   */
  TVM_DLL virtual Array<AttrFieldInfo> ListFieldInfo() const = 0;
  /*!
   * \brief Initialize the attributes by arguments.
   * \param kwargs The key value pairs for initialization.
   *        [key0, value0, key1, value1, ..., key_n, value_n]
   * \param allow_unknown Whether allow additional unknown fields.
   * \note This function throws when the required field is not present.
   */
  TVM_DLL virtual void InitByPackedArgs(const TVMArgs& kwargs, bool allow_unknown = false) = 0;

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const char* _type_key = "Attrs";
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
  Map<String, ObjectRef> dict;

  bool SEqualReduce(const DictAttrsNode* other, SEqualReducer equal) const {
    return equal(dict, other->dict);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(dict); }

  // implementations
  void VisitAttrs(AttrVisitor* v) final;
  void VisitNonDefaultAttrs(AttrVisitor* v) final;
  void InitByPackedArgs(const runtime::TVMArgs& args, bool allow_unknown) final;
  Array<AttrFieldInfo> ListFieldInfo() const final;

  // type info
  static constexpr const char* _type_key = "DictAttrs";
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
  TVM_DLL explicit DictAttrs(Map<String, ObjectRef> dict);

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
      Optional<TObjectRef> default_value = Optional<TObjectRef>(nullptr)) const {
    static_assert(std::is_base_of<ObjectRef, TObjectRef>::value,
                  "Can only call GetAttr with ObjectRef types.");
    if (!defined()) return default_value;
    const DictAttrsNode* node = this->as<DictAttrsNode>();

    auto it = node->dict.find(attr_key);
    if (it != node->dict.end()) {
      return Downcast<Optional<TObjectRef>>((*it).second);
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

  TVM_DEFINE_OBJECT_REF_METHODS(DictAttrs, Attrs, DictAttrsNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DictAttrsNode);
};

/*!
 * \brief Create an Attr object with all default values.
 * \tparam TAttrNode the type to be created.
 * \return A instance that will represent None.
 */
template <typename TAttrs>
inline TAttrs AttrsWithDefaultValues() {
  static_assert(std::is_base_of<Attrs, TAttrs>::value, "Can only take attr nodes");
  auto n = make_object<typename TAttrs::ContainerType>();
  n->InitByPackedArgs(runtime::TVMArgs(nullptr, nullptr, 0), false);
  return TAttrs(n);
}

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
inline TFunc WithAttr(TFunc input, const std::string& attr_key, ObjectRef attr_value) {
  using TNode = typename TFunc::ContainerType;
  static_assert(TNode::_type_final, "Can only operate on the leaf nodes");
  TNode* node = input.CopyOnWrite();
  if (node->attrs.defined()) {
    node->attrs.CopyOnWrite()->dict.Set(attr_key, attr_value);
  } else {
    Map<String, ObjectRef> dict = {{attr_key, attr_value}};
    node->attrs = DictAttrs(dict);
  }
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
inline TFunc WithAttrs(TFunc input, Map<String, ObjectRef> attrs) {
  using TNode = typename TFunc::ContainerType;
  static_assert(TNode::_type_final, "Can only operate on the leaf nodes");
  TNode* node = input.CopyOnWrite();
  if (node->attrs.defined()) {
    for (const auto& pair : attrs) {
      node->attrs.CopyOnWrite()->dict.Set(pair.first, pair.second);
    }
  } else {
    node->attrs = DictAttrs(std::move(attrs));
  }
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

  if (input->attrs.defined()) {
    TNode* node = input.CopyOnWrite();
    node->attrs.CopyOnWrite()->dict.erase(attr_key);
    if (node->attrs->dict.size() == 0) {
      node->attrs = NullValue<DictAttrs>();
    }
  }
  return input;
}

// Namespace containing detail implementations
namespace detail {
using runtime::TVMArgValue;

// helper entry that does nothing in set_default/bound/describe calls.
struct AttrNopEntry {
  using TSelf = AttrNopEntry;

  TSelf& describe(DMLC_ATTRIBUTE_UNUSED const char* str) { return *this; }
  template <typename T>
  TSelf& set_default(DMLC_ATTRIBUTE_UNUSED const T& value) {
    return *this;
  }
  template <typename T>
  TSelf& set_lower_bound(DMLC_ATTRIBUTE_UNUSED const T& begin) {
    return *this;
  }
  template <typename T>
  TSelf& set_upper_bound(DMLC_ATTRIBUTE_UNUSED const T& end) {
    return *this;
  }
};

// Wrapper for normal visitor.
class AttrNormalVisitor {
 public:
  explicit AttrNormalVisitor(AttrVisitor* visitor) : visitor_(visitor) {}
  template <typename T>
  AttrNopEntry operator()(const char* key, T* value) {
    visitor_->Visit(key, value);
    return AttrNopEntry();
  }

 private:
  AttrVisitor* visitor_;
};

class AttrsSEqualVisitor {
 public:
  bool result_{true};
  // constructor
  AttrsSEqualVisitor(const Object* lhs, const Object* rhs, const SEqualReducer& equal)
      : lhs_(lhs), rhs_(rhs), equal_(equal) {}
  template <typename T>
  AttrNopEntry operator()(const char* key, T* lhs_value) {
    if (!result_) return AttrNopEntry();
    const T* rhs_value = reinterpret_cast<const T*>(
        reinterpret_cast<const char*>(rhs_) +
        (reinterpret_cast<const char*>(lhs_value) - reinterpret_cast<const char*>(lhs_)));
    if (!equal_(*lhs_value, *rhs_value)) {
      result_ = false;
    }
    return AttrNopEntry();
  }

 private:
  const Object* lhs_;
  const Object* rhs_;
  const SEqualReducer& equal_;
};

class AttrsSHashVisitor {
 public:
  explicit AttrsSHashVisitor(const SHashReducer& hash_reducer) : hash_reducer_(hash_reducer) {}

  template <typename T>
  AttrNopEntry operator()(const char* key, T* value) {
    hash_reducer_(*value);
    return AttrNopEntry();
  }

 private:
  const SHashReducer& hash_reducer_;
};

// helper entry that does initialization, set default.
template <typename T>
struct AttrInitEntry {
  // The attributes
  using TSelf = AttrInitEntry<T>;
  // The type key
  const char* type_key_;
  // field name
  const char* key_;
  // internal value.
  T* value_;
  // whether the value is missing.
  // NOTE: initialize to false so that the destructor does not throw unless
  // AttrInitVisitor::operator() is committed to returning an instance of this class.
  // It is expected not to set this to true until that is true.
  bool value_missing_{false};

  AttrInitEntry() = default;

  AttrInitEntry(AttrInitEntry&& other) {
    type_key_ = other.type_key_;
    key_ = other.key_;
    value_ = other.value_;
    value_missing_ = other.value_missing_;
    // avoid unexpected throw
    other.value_missing_ = false;
  }

  // If the value is still missing in destruction time throw an error.
  ~AttrInitEntry() DMLC_THROW_EXCEPTION {
    if (value_missing_) {
      std::ostringstream os;
      os << type_key_ << ": Cannot find required field \'" << key_ << "\' during initialization. "
         << "If the key is defined check that its type matches the declared type.";
      throw AttrError(os.str());
    }
  }
  // override fields.
  // This function sets the lower bound of the attribute
  TSelf& set_lower_bound(const T& begin) {
    if (this->value_missing_) return *this;
    const T& val = *value_;
    if (begin > val) {
      std::ostringstream os;
      os << type_key_ << "." << key_ << ": "
         << "value " << val << " is smaller than the lower bound " << begin;
      throw AttrError(os.str());
    }
    return *this;
  }
  // This function sets the upper bound of the attribute
  TSelf& set_upper_bound(const T& end) {
    if (this->value_missing_) return *this;
    const T& val = *value_;
    if (val > end) {
      std::ostringstream os;
      os << type_key_ << "." << key_ << ": "
         << "value " << val << " is bigger than the upper bound " << end;
      throw AttrError(os.str());
    }
    return *this;
  }
  // set default when
  TSelf& set_default(const T& value) {
    if (!value_missing_) return *this;
    *value_ = value;
    value_missing_ = false;
    return *this;
  }
  TSelf& describe(DMLC_ATTRIBUTE_UNUSED const char* str) { return *this; }
};

// Template function to allow smart conversion
// from Expr types into the constants.
template <typename T>
inline void SetValue(T* ptr, const TVMArgValue& val) {
  *ptr = val.operator T();
}

template <typename T>
inline void SetIntValue(T* ptr, const TVMArgValue& val) {
  if (val.type_code() == kDLInt) {
    *ptr = static_cast<T>(val.value().v_int64);
  } else {
    IntImm expr = val;
    *ptr = static_cast<T>(expr->value);
  }
}

// Workaround for GCC8.1 / GCC8.2
template <>
inline void SetValue<DataType>(DataType* ptr, const TVMArgValue& val) {
  *ptr = val.operator DataType();
}

template <>
inline void SetValue<std::string>(std::string* ptr, const TVMArgValue& val) {
  if (String::CanConvertFrom(val)) {
    *ptr = val.operator std::string();
  } else {
    LOG(FATAL) << "Expect str";
  }
}

template <>
inline void SetValue<double>(double* ptr, const TVMArgValue& val) {
  if (val.type_code() == kDLFloat || val.type_code() == kDLInt) {
    *ptr = val.operator double();
  } else {
    ObjectRef expr = val;
    ICHECK(expr.defined());
    if (const IntImmNode* op = expr.as<IntImmNode>()) {
      *ptr = static_cast<double>(op->value);
    } else if (const FloatImmNode* op = expr.as<FloatImmNode>()) {
      *ptr = static_cast<double>(op->value);
    } else {
      LOG(FATAL) << "Expect float value, but get " << expr->GetTypeKey();
    }
  }
}
template <>
inline void SetValue<int>(int* ptr, const TVMArgValue& val) {
  SetIntValue(ptr, val);
}
template <>
inline void SetValue<int64_t>(int64_t* ptr, const TVMArgValue& val) {
  SetIntValue(ptr, val);
}
template <>
inline void SetValue<uint64_t>(uint64_t* ptr, const TVMArgValue& val) {
  SetIntValue(ptr, val);
}
template <>
inline void SetValue<bool>(bool* ptr, const TVMArgValue& val) {
  SetIntValue(ptr, val);
}

// Visitor for value initialization
template <typename FFind>
class AttrInitVisitor {
 public:
  // Counter of number of matched attributes during visit.
  // This is used to decide if there is additional unmatched attributes.
  size_t hit_count_{0};
  // constructor
  AttrInitVisitor(const char* type_key, FFind ffind) : type_key_(type_key), ffind_(ffind) {}

  template <typename T>
  AttrInitEntry<T> operator()(const char* key, T* value) {
    TVMArgValue val;
    AttrInitEntry<T> opt;
    opt.type_key_ = type_key_;
    opt.key_ = key;
    opt.value_ = value;
    if (ffind_(key, &val)) {
      SetValue(value, val);
      opt.value_missing_ = false;
      ++hit_count_;
    } else {
      opt.value_missing_ = true;
    }
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wpessimizing-move"
#endif
    return std::move(opt);
  }

 private:
  // the type key
  const char* type_key_;
  FFind ffind_;
};

template <typename FFind>
inline AttrInitVisitor<FFind> CreateInitVisitor(const char* type_key, FFind ffind) {
  return AttrInitVisitor<FFind>(type_key, ffind);
}

/*!
 * \brief Helper struct to get the type name known to tvm.
 * \tparam T the type we are interested in.
 */
template <typename T>
struct TypeName {
  static constexpr const char* value = T::ContainerType::_type_key;
};

template <>
struct TypeName<int> {
  static constexpr const char* value = "int";
};

template <>
struct TypeName<int64_t> {
  static constexpr const char* value = "int64";
};

template <>
struct TypeName<uint64_t> {
  static constexpr const char* value = "uint64_t";
};

template <>
struct TypeName<DataType> {
  static constexpr const char* value = "DataType";
};

template <>
struct TypeName<std::string> {
  static constexpr const char* value = "str";
};

template <>
struct TypeName<bool> {
  static constexpr const char* value = "bool";
};

template <>
struct TypeName<void*> {
  static constexpr const char* value = "handle";
};

template <>
struct TypeName<double> {
  static constexpr const char* value = "double";
};

class AttrDocEntry {
 public:
  using TSelf = AttrDocEntry;

  explicit AttrDocEntry(ObjectPtr<AttrFieldInfoNode> info) : info_(info) {}
  TSelf& describe(const char* str) {
    info_->description = str;
    return *this;
  }
  template <typename T>
  TSelf& set_default(const T& value) {
    std::ostringstream os;
    os << info_->type_info << ", default=" << value;
    info_->type_info = os.str();
    return *this;
  }
  template <typename T>
  TSelf& set_lower_bound(DMLC_ATTRIBUTE_UNUSED T begin) {
    return *this;
  }
  template <typename T>
  TSelf& set_upper_bound(DMLC_ATTRIBUTE_UNUSED T end) {
    return *this;
  }

 private:
  ObjectPtr<AttrFieldInfoNode> info_;
};

class AttrDocVisitor {
 public:
  template <typename T>
  AttrDocEntry operator()(const char* key, T* v) {
    ObjectPtr<AttrFieldInfoNode> info = make_object<AttrFieldInfoNode>();
    info->name = key;
    info->type_info = TypeName<T>::value;
    fields_.push_back(AttrFieldInfo(info));
    return AttrDocEntry(info);
  }

  Array<AttrFieldInfo> fields_;
};

class AttrExistVisitor {
 public:
  std::string key_;
  bool exist_{false};

  template <typename T>
  AttrNopEntry operator()(const char* key, T* v) {
    if (exist_) return AttrNopEntry();
    if (key == key_) exist_ = true;
    return AttrNopEntry();
  }
};

template <typename T>
struct AttrTriggerNonDefaultEntry {
  using TSelf = AttrTriggerNonDefaultEntry<T>;
  // constructor
  AttrTriggerNonDefaultEntry(AttrVisitor* visitor, const char* key, T* data)
      : visitor_(visitor), key_(key), data_(data) {}

  ~AttrTriggerNonDefaultEntry() DMLC_THROW_EXCEPTION {
    if (trigger_) {
      visitor_->Visit(key_, data_);
    }
  }
  TSelf& describe(DMLC_ATTRIBUTE_UNUSED const char* str) { return *this; }
  TSelf& set_default(const T& value) {
    if (tvm::StructuralEqual()(value, *data_)) {
      trigger_ = false;
    }
    return *this;
  }
  TSelf& set_lower_bound(DMLC_ATTRIBUTE_UNUSED const T& begin) { return *this; }
  TSelf& set_upper_bound(DMLC_ATTRIBUTE_UNUSED const T& end) { return *this; }

 private:
  AttrVisitor* visitor_;
  const char* key_;
  T* data_;
  bool trigger_{true};
};

class AttrNonDefaultVisitor {
 public:
  explicit AttrNonDefaultVisitor(AttrVisitor* visitor) : visitor_(visitor) {}
  template <typename T>
  AttrTriggerNonDefaultEntry<T> operator()(const char* key, T* value) {
    return AttrTriggerNonDefaultEntry<T>(visitor_, key, value);
  }

 private:
  AttrVisitor* visitor_;
};
}  // namespace detail

/*!
 * \brief The base class of the all the
 *  Use "curiously recurring template pattern".
 *
 * \tparam DerivedType The final attribute type.
 */
template <typename DerivedType>
class AttrsNode : public BaseAttrsNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    ::tvm::detail::AttrNormalVisitor vis(v);
    self()->_tvm_VisitAttrs(vis);
  }

  void VisitNonDefaultAttrs(AttrVisitor* v) {
    ::tvm::detail::AttrNonDefaultVisitor vis(v);
    self()->_tvm_VisitAttrs(vis);
  }

  void InitByPackedArgs(const runtime::TVMArgs& args, bool allow_unknown) final {
    ICHECK_EQ(args.size() % 2, 0);
    const int kLinearSearchBound = 16;
    int hit_count = 0;
    // applies two strategies to lookup
    if (args.size() < kLinearSearchBound) {
      // linear search.
      auto ffind = [&args](const char* key, runtime::TVMArgValue* val) {
        for (int i = 0; i < args.size(); i += 2) {
          ICHECK_EQ(args.type_codes[i], kTVMStr);
          if (!std::strcmp(key, args.values[i].v_str)) {
            *val = args[i + 1];
            return true;
          }
        }
        return false;
      };
      auto vis = ::tvm::detail::CreateInitVisitor(DerivedType::_type_key, ffind);
      self()->_tvm_VisitAttrs(vis);
      hit_count = vis.hit_count_;
    } else {
      // construct a map then do lookup.
      std::unordered_map<std::string, runtime::TVMArgValue> kwargs;
      for (int i = 0; i < args.size(); i += 2) {
        ICHECK_EQ(args.type_codes[i], kTVMStr);
        kwargs[args[i].operator std::string()] = args[i + 1];
      }
      auto ffind = [&kwargs](const char* key, runtime::TVMArgValue* val) {
        auto it = kwargs.find(key);
        if (it != kwargs.end()) {
          *val = it->second;
          return true;
        }
        return false;
      };
      auto vis = ::tvm::detail::CreateInitVisitor(DerivedType::_type_key, ffind);
      self()->_tvm_VisitAttrs(vis);
      hit_count = vis.hit_count_;
    }
    // error handling, slow path
    if (hit_count * 2 != args.size() && !allow_unknown) {
      for (int i = 0; i < args.size(); i += 2) {
        ::tvm::detail::AttrExistVisitor visitor;
        visitor.key_ = args[i].operator std::string();
        self()->_tvm_VisitAttrs(visitor);
        if (!visitor.exist_) {
          std::ostringstream os;
          os << DerivedType::_type_key << ": does not have field \'" << visitor.key_
             << "\', Possible fields:\n";
          os << "----------------\n";
          this->PrintDocString(os);
          throw AttrError(os.str());
        }
      }
    }
  }

  bool SEqualReduce(const DerivedType* other, SEqualReducer equal) const {
    DerivedType* pself = self();
    ::tvm::detail::AttrsSEqualVisitor visitor(pself, other, equal);
    self()->_tvm_VisitAttrs(visitor);
    return visitor.result_;
  }

  void SHashReduce(SHashReducer hash_reducer) const {
    ::tvm::detail::AttrsSHashVisitor visitor(hash_reducer);
    self()->_tvm_VisitAttrs(visitor);
  }

  Array<AttrFieldInfo> ListFieldInfo() const final {
    ::tvm::detail::AttrDocVisitor visitor;
    self()->_tvm_VisitAttrs(visitor);
    return visitor.fields_;
  }

 private:
  DerivedType* self() const {
    return const_cast<DerivedType*>(static_cast<const DerivedType*>(this));
  }
};

template <typename... Args>
inline void BaseAttrsNode::InitBySeq(Args&&... args) {
  runtime::PackedFunc pf(
      [this](const TVMArgs& args, TVMRetValue* rv) { this->InitByPackedArgs(args); });
  pf(std::forward<Args>(args)...);
}

inline void BaseAttrsNode::PrintDocString(std::ostream& os) const {  // NOLINT(*)
  Array<AttrFieldInfo> entry = this->ListFieldInfo();
  for (AttrFieldInfo info : entry) {
    os << info->name << " : " << info->type_info << '\n';
    if (info->description.length() != 0) {
      os << "    " << info->description << '\n';
    }
  }
}

}  // namespace tvm
#endif  // TVM_IR_ATTRS_H_
