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
 * \file tvm/ffi/reflection/reflection.h
 * \brief Base reflection support to access object fields.
 */
#ifndef TVM_FFI_REFLECTION_REFLECTION_H_
#define TVM_FFI_REFLECTION_REFLECTION_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/type_traits.h>

#include <string>
#include <utility>

namespace tvm {
namespace ffi {
/*! \brief Reflection namespace */
namespace reflection {

/*! \brief Trait that can be used to set field info */
struct FieldInfoTrait {};

/*!
 * \brief Trait that can be used to set field default value
 */
class DefaultValue : public FieldInfoTrait {
 public:
  explicit DefaultValue(Any value) : value_(value) {}

  TVM_FFI_INLINE void Apply(TVMFFIFieldInfo* info) const {
    info->default_value = AnyView(value_).CopyToTVMFFIAny();
    info->flags |= kTVMFFIFieldFlagBitMaskHasDefault;
  }

 private:
  Any value_;
};

/*!
 * \brief Get the byte offset of a class member field.
 *
 * \tparam The original class.
 * \tparam T the field type.
 *
 * \param field_ptr A class member pointer
 * \returns The byteoffset
 */
template <typename Class, typename T>
TVM_FFI_INLINE int64_t GetFieldByteOffsetToObject(T Class::*field_ptr) {
  int64_t field_offset_to_class =
      reinterpret_cast<int64_t>(&(static_cast<Class*>(nullptr)->*field_ptr));
  return field_offset_to_class - details::ObjectUnsafe::GetObjectOffsetToSubclass<Class>();
}

class ReflectionDefBase {
 protected:
  template <typename T>
  static int FieldGetter(void* field, TVMFFIAny* result) {
    TVM_FFI_SAFE_CALL_BEGIN();
    *result = details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(*reinterpret_cast<T*>(field)));
    TVM_FFI_SAFE_CALL_END();
  }

  template <typename T>
  static int FieldSetter(void* field, const TVMFFIAny* value) {
    TVM_FFI_SAFE_CALL_BEGIN();
    *reinterpret_cast<T*>(field) = AnyView::CopyFromTVMFFIAny(*value).cast<T>();
    TVM_FFI_SAFE_CALL_END();
  }

  template <typename T>
  static int ObjectCreatorDefault(TVMFFIObjectHandle* result) {
    TVM_FFI_SAFE_CALL_BEGIN();
    ObjectPtr<T> obj = make_object<T>();
    *result = details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(obj));
    TVM_FFI_SAFE_CALL_END();
  }

  template <typename T>
  TVM_FFI_INLINE static void ApplyFieldInfoTrait(TVMFFIFieldInfo* info, const T& value) {
    if constexpr (std::is_base_of_v<FieldInfoTrait, std::decay_t<T>>) {
      value.Apply(info);
    }
    if constexpr (std::is_same_v<std::decay_t<T>, char*>) {
      info->doc = TVMFFIByteArray{value, std::char_traits<char>::length(value)};
    }
  }

  template <typename T>
  TVM_FFI_INLINE static void ApplyMethodInfoTrait(TVMFFIMethodInfo* info, const T& value) {
    if constexpr (std::is_same_v<std::decay_t<T>, char*>) {
      info->doc = TVMFFIByteArray{value, std::char_traits<char>::length(value)};
    }
  }

  template <typename T>
  TVM_FFI_INLINE static void ApplyExtraInfoTrait(TVMFFITypeExtraInfo* info, const T& value) {
    if constexpr (std::is_same_v<std::decay_t<T>, char*>) {
      info->doc = TVMFFIByteArray{value, std::char_traits<char>::length(value)};
    }
  }

  template <typename Class, typename R, typename... Args>
  TVM_FFI_INLINE static Function GetMethod(std::string name, R (Class::*func)(Args...)) {
    static_assert(std::is_base_of_v<ObjectRef, Class> || std::is_base_of_v<Object, Class>,
                  "Class must be derived from ObjectRef or Object");
    if constexpr (std::is_base_of_v<ObjectRef, Class>) {
      auto fwrap = [func](Class target, Args... params) -> R {
        // call method pointer
        return (target.*func)(std::forward<Args>(params)...);
      };
      return ffi::Function::FromTyped(fwrap, name);
    }

    if constexpr (std::is_base_of_v<Object, Class>) {
      auto fwrap = [func](const Class* target, Args... params) -> R {
        // call method pointer
        return (const_cast<Class*>(target)->*func)(std::forward<Args>(params)...);
      };
      return ffi::Function::FromTyped(fwrap, name);
    }
  }

  template <typename Class, typename R, typename... Args>
  TVM_FFI_INLINE static Function GetMethod(std::string name, R (Class::*func)(Args...) const) {
    static_assert(std::is_base_of_v<ObjectRef, Class> || std::is_base_of_v<Object, Class>,
                  "Class must be derived from ObjectRef or Object");
    if constexpr (std::is_base_of_v<ObjectRef, Class>) {
      auto fwrap = [func](const Class target, Args... params) -> R {
        // call method pointer
        return (target.*func)(std::forward<Args>(params)...);
      };
      return ffi::Function::FromTyped(fwrap, name);
    }

    if constexpr (std::is_base_of_v<Object, Class>) {
      auto fwrap = [func](const Class* target, Args... params) -> R {
        // call method pointer
        return (target->*func)(std::forward<Args>(params)...);
      };
      return ffi::Function::FromTyped(fwrap, name);
    }
  }

  template <typename Class, typename Func>
  TVM_FFI_INLINE static Function GetMethod(std::string name, Func&& func) {
    return ffi::Function::FromTyped(std::forward<Func>(func), name);
  }
};

class GlobalDef : public ReflectionDefBase {
 public:
  /*
   * \brief Define a global function.
   *
   * \tparam Func The function type.
   * \tparam Extra The extra arguments.
   *
   * \param name The name of the function.
   * \param func The function to be registered.
   * \param extra The extra arguments that can be docstring.
   *
   * \return The reflection definition.
   */
  template <typename Func, typename... Extra>
  GlobalDef& def(const char* name, Func&& func, Extra&&... extra) {
    RegisterFunc(name, ffi::Function::FromTyped(std::forward<Func>(func), std::string(name)),
                 std::forward<Extra>(extra)...);
    return *this;
  }

  /*
   * \brief Define a global function in ffi::PackedArgs format.
   *
   * \tparam Func The function type.
   * \tparam Extra The extra arguments.
   *
   * \param name The name of the function.
   * \param func The function to be registered.
   * \param extra The extra arguments that can be docstring.
   *
   * \return The reflection definition.
   */
  template <typename Func, typename... Extra>
  GlobalDef& def_packed(const char* name, Func func, Extra&&... extra) {
    RegisterFunc(name, ffi::Function::FromPacked(func), std::forward<Extra>(extra)...);
    return *this;
  }

  /*
   * \brief Expose a class method as a global function.
   *
   * An argument will be added to the first position if the function is not static.
   *
   * \tparam Class The class type.
   * \tparam Func The function type.
   *
   * \param name The name of the method.
   * \param func The function to be registered.
   *
   * \return The reflection definition.
   */
  template <typename Func, typename... Extra>
  GlobalDef& def_method(const char* name, Func&& func, Extra&&... extra) {
    RegisterFunc(name, GetMethod_(std::string(name), std::forward<Func>(func)),
                 std::forward<Extra>(extra)...);
    return *this;
  }

 private:
  template <typename Func>
  TVM_FFI_INLINE static Function GetMethod_(std::string name, Func&& func) {
    return ffi::Function::FromTyped(std::forward<Func>(func), name);
  }

  template <typename Class, typename R, typename... Args>
  TVM_FFI_INLINE static Function GetMethod_(std::string name, R (Class::*func)(Args...) const) {
    return GetMethod<Class>(std::string(name), func);
  }

  template <typename Class, typename R, typename... Args>
  TVM_FFI_INLINE static Function GetMethod_(std::string name, R (Class::*func)(Args...)) {
    return GetMethod<Class>(std::string(name), func);
  }

  template <typename... Extra>
  void RegisterFunc(const char* name, ffi::Function func, Extra&&... extra) {
    TVMFFIMethodInfo info;
    info.name = TVMFFIByteArray{name, std::char_traits<char>::length(name)};
    info.doc = TVMFFIByteArray{nullptr, 0};
    info.type_schema = TVMFFIByteArray{nullptr, 0};
    info.flags = 0;
    // obtain the method function
    info.method = AnyView(func).CopyToTVMFFIAny();
    // apply method info traits
    ((ApplyMethodInfoTrait(&info, std::forward<Extra>(extra)), ...));
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIFunctionSetGlobalFromMethodInfo(&info, 0));
  }
};

template <typename Class>
class ObjectDef : public ReflectionDefBase {
 public:
  template <typename... ExtraArgs>
  explicit ObjectDef(ExtraArgs&&... extra_args)
      : type_index_(Class::_GetOrAllocRuntimeTypeIndex()), type_key_(Class::_type_key) {
    RegisterExtraInfo(std::forward<ExtraArgs>(extra_args)...);
  }

  /*!
   * \brief Define a readonly field.
   *
   * \tparam Class The class type.
   * \tparam T The field type.
   * \tparam Extra The extra arguments.
   *
   * \param name The name of the field.
   * \param field_ptr The pointer to the field.
   * \param extra The extra arguments that can be docstring or default value.
   *
   * \return The reflection definition.
   */
  template <typename T, typename BaseClass, typename... Extra>
  TVM_FFI_INLINE ObjectDef& def_ro(const char* name, T BaseClass::*field_ptr, Extra&&... extra) {
    RegisterField(name, field_ptr, false, std::forward<Extra>(extra)...);
    return *this;
  }

  /*!
   * \brief Define a read-write field.
   *
   * \tparam Class The class type.
   * \tparam T The field type.
   * \tparam Extra The extra arguments.
   *
   * \param name The name of the field.
   * \param field_ptr The pointer to the field.
   * \param extra The extra arguments that can be docstring or default value.
   *
   * \return The reflection definition.
   */
  template <typename T, typename BaseClass, typename... Extra>
  TVM_FFI_INLINE ObjectDef& def_rw(const char* name, T BaseClass::*field_ptr, Extra&&... extra) {
    static_assert(Class::_type_mutable, "Only mutable classes are supported for writable fields");
    RegisterField(name, field_ptr, true, std::forward<Extra>(extra)...);
    return *this;
  }

  /*!
   * \brief Define a method.
   *
   * \tparam Func The function type.
   * \tparam Extra The extra arguments.
   *
   * \param name The name of the method.
   * \param func The function to be registered.
   * \param extra The extra arguments that can be docstring.
   *
   * \return The reflection definition.
   */
  template <typename Func, typename... Extra>
  TVM_FFI_INLINE ObjectDef& def(const char* name, Func&& func, Extra&&... extra) {
    RegisterMethod(name, false, std::forward<Func>(func), std::forward<Extra>(extra)...);
    return *this;
  }

  /*!
   * \brief Define a static method.
   *
   * \tparam Func The function type.
   * \tparam Extra The extra arguments.
   *
   * \param name The name of the method.
   * \param func The function to be registered.
   * \param extra The extra arguments that can be docstring.
   *
   * \return The reflection definition.
   */
  template <typename Func, typename... Extra>
  TVM_FFI_INLINE ObjectDef& def_static(const char* name, Func&& func, Extra&&... extra) {
    RegisterMethod(name, true, std::forward<Func>(func), std::forward<Extra>(extra)...);
    return *this;
  }

 private:
  template <typename... ExtraArgs>
  void RegisterExtraInfo(ExtraArgs&&... extra_args) {
    TVMFFITypeExtraInfo info;
    info.total_size = sizeof(Class);
    info.creator = nullptr;
    info.doc = TVMFFIByteArray{nullptr, 0};
    if constexpr (std::is_default_constructible_v<Class>) {
      info.creator = ObjectCreatorDefault<Class>;
    }
    // apply extra info traits
    ((ApplyExtraInfoTrait(&info, std::forward<ExtraArgs>(extra_args)), ...));
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterExtraInfo(type_index_, &info));
  }

  template <typename T, typename BaseClass, typename... ExtraArgs>
  void RegisterField(const char* name, T BaseClass::*field_ptr, bool writable,
                     ExtraArgs&&... extra_args) {
    static_assert(std::is_base_of_v<BaseClass, Class>, "BaseClass must be a base class of Class");
    TVMFFIFieldInfo info;
    info.name = TVMFFIByteArray{name, std::char_traits<char>::length(name)};
    info.field_static_type_index = TypeToFieldStaticTypeIndex<T>::value;
    // store byte offset and setter, getter
    // so the same setter can be reused for all the same type
    info.offset = GetFieldByteOffsetToObject<Class, T>(field_ptr);
    info.size = sizeof(T);
    info.alignment = alignof(T);
    info.flags = 0;
    if (writable) {
      info.flags |= kTVMFFIFieldFlagBitMaskWritable;
    }
    info.getter = FieldGetter<T>;
    info.setter = FieldSetter<T>;
    // initialize default value to nullptr
    info.default_value = AnyView(nullptr).CopyToTVMFFIAny();
    info.doc = TVMFFIByteArray{nullptr, 0};
    info.type_schema = TVMFFIByteArray{nullptr, 0};
    // apply field info traits
    ((ApplyFieldInfoTrait(&info, std::forward<ExtraArgs>(extra_args)), ...));
    // call register
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterField(type_index_, &info));
  }

  // register a method
  template <typename Func, typename... Extra>
  void RegisterMethod(const char* name, bool is_static, Func&& func, Extra&&... extra) {
    TVMFFIMethodInfo info;
    info.name = TVMFFIByteArray{name, std::char_traits<char>::length(name)};
    info.doc = TVMFFIByteArray{nullptr, 0};
    info.type_schema = TVMFFIByteArray{nullptr, 0};
    info.flags = 0;
    if (is_static) {
      info.flags |= kTVMFFIFieldFlagBitMaskIsStaticMethod;
    }
    // obtain the method function
    Function method =
        GetMethod<Class>(std::string(type_key_) + "." + name, std::forward<Func>(func));
    info.method = AnyView(method).CopyToTVMFFIAny();
    // apply method info traits
    ((ApplyMethodInfoTrait(&info, std::forward<Extra>(extra)), ...));
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterMethod(type_index_, &info));
  }

  int32_t type_index_;
  const char* type_key_;
};

/*!
 * \brief helper function to get reflection field info by type key and field name
 */
inline const TVMFFIFieldInfo* GetFieldInfo(std::string_view type_key, const char* field_name) {
  int32_t type_index;
  TVMFFIByteArray type_key_array = {type_key.data(), type_key.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  const TypeInfo* info = TVMFFIGetTypeInfo(type_index);
  for (int32_t i = 0; i < info->num_fields; ++i) {
    if (std::strncmp(info->fields[i].name.data, field_name, info->fields[i].name.size) == 0) {
      return &(info->fields[i]);
    }
  }
  TVM_FFI_THROW(RuntimeError) << "Cannot find field " << field_name << " in " << type_key;
  TVM_FFI_UNREACHABLE();
}

/*!
 * \brief helper wrapper class to obtain a getter.
 */
class FieldGetter {
 public:
  explicit FieldGetter(const TVMFFIFieldInfo* field_info) : field_info_(field_info) {}

  explicit FieldGetter(std::string_view type_key, const char* field_name)
      : FieldGetter(GetFieldInfo(type_key, field_name)) {}

  Any operator()(const Object* obj_ptr) const {
    Any result;
    const void* addr = reinterpret_cast<const char*>(obj_ptr) + field_info_->offset;
    TVM_FFI_CHECK_SAFE_CALL(
        field_info_->getter(const_cast<void*>(addr), reinterpret_cast<TVMFFIAny*>(&result)));
    return result;
  }

  Any operator()(const ObjectPtr<Object>& obj_ptr) const { return operator()(obj_ptr.get()); }

  Any operator()(const ObjectRef& obj) const { return operator()(obj.get()); }

 private:
  const TVMFFIFieldInfo* field_info_;
};

/*!
 * \brief helper wrapper class to obtain a setter.
 */
class FieldSetter {
 public:
  explicit FieldSetter(const TVMFFIFieldInfo* field_info) : field_info_(field_info) {}

  explicit FieldSetter(std::string_view type_key, const char* field_name)
      : FieldSetter(GetFieldInfo(type_key, field_name)) {}

  void operator()(const Object* obj_ptr, AnyView value) const {
    const void* addr = reinterpret_cast<const char*>(obj_ptr) + field_info_->offset;
    TVM_FFI_CHECK_SAFE_CALL(
        field_info_->setter(const_cast<void*>(addr), reinterpret_cast<const TVMFFIAny*>(&value)));
  }

  void operator()(const ObjectPtr<Object>& obj_ptr, AnyView value) const {
    operator()(obj_ptr.get(), value);
  }

  void operator()(const ObjectRef& obj, AnyView value) const { operator()(obj.get(), value); }

 private:
  const TVMFFIFieldInfo* field_info_;
};

/*!
 * \brief helper function to get reflection method info by type key and method name
 *
 * \param type_key The type key.
 * \param method_name The name of the method.
 * \return The method info.
 */
inline const TVMFFIMethodInfo* GetMethodInfo(std::string_view type_key, const char* method_name) {
  int32_t type_index;
  TVMFFIByteArray type_key_array = {type_key.data(), type_key.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  const TypeInfo* info = TVMFFIGetTypeInfo(type_index);
  for (int32_t i = 0; i < info->num_methods; ++i) {
    if (std::strncmp(info->methods[i].name.data, method_name, info->methods[i].name.size) == 0) {
      return &(info->methods[i]);
    }
  }
  TVM_FFI_THROW(RuntimeError) << "Cannot find method " << method_name << " in " << type_key;
  TVM_FFI_UNREACHABLE();
}

/*!
 * \brief helper function to get reflection method function by method info
 *
 * \param type_key The type key.
 * \param method_name The name of the method.
 * \return The method function.
 */
inline Function GetMethod(std::string_view type_key, const char* method_name) {
  const TVMFFIMethodInfo* info = GetMethodInfo(type_key, method_name);
  return AnyView::CopyFromTVMFFIAny(info->method).cast<Function>();
}

/*!
 * \brief Visit each field info of the type info and run callback.
 *
 * \tparam Callback The callback function type.
 *
 * \param type_info The type info.
 * \param callback The callback function.
 *
 * \note This function calls both the child and parent type info.
 */
template <typename Callback>
inline void ForEachFieldInfo(const TypeInfo* type_info, Callback callback) {
  using ResultType = decltype(callback(type_info->fields));
  static_assert(std::is_same_v<ResultType, void>, "Callback must return void");
  // iterate through acenstors in parent to child order
  // skip the first one since it is always the root object
  for (int i = 1; i < type_info->type_depth; ++i) {
    const TVMFFITypeInfo* parent_info = type_info->type_acenstors[i];
    for (int j = 0; j < parent_info->num_fields; ++j) {
      callback(parent_info->fields + j);
    }
  }
  for (int i = 0; i < type_info->num_fields; ++i) {
    callback(type_info->fields + i);
  }
}

/*!
 * \brief Visit each field info of the type info and run callback which returns bool for early stop.
 *
 * \tparam Callback The callback function type, which returns bool for early stop.
 *
 * \param type_info The type info.
 * \param callback_with_early_stop The callback function.
 * \return true if any of early stop is triggered.
 *
 * \note This function calls both the child and parent type info and can be used for searching.
 */
template <typename Callback>
inline bool ForEachFieldInfoWithEarlyStop(const TypeInfo* type_info,
                                          Callback callback_with_early_stop) {
  // iterate through acenstors in parent to child order
  // skip the first one since it is always the root object
  for (int i = 1; i < type_info->type_depth; ++i) {
    const TVMFFITypeInfo* parent_info = type_info->type_acenstors[i];
    for (int j = 0; j < parent_info->num_fields; ++j) {
      if (callback_with_early_stop(parent_info->fields + j)) return true;
    }
  }
  for (int i = 0; i < type_info->num_fields; ++i) {
    if (callback_with_early_stop(type_info->fields + i)) return true;
  }
  return false;
}

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_REFLECTION_REFLECTION_H_
