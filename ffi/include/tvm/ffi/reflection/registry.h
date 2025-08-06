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
 * \file tvm/ffi/reflection/registry.h
 * \brief Registry of reflection metadata.
 */
#ifndef TVM_FFI_REFLECTION_REGISTRY_H_
#define TVM_FFI_REFLECTION_REGISTRY_H_

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

/*
 * \brief Trait that can be used to attach field flag
 */
class AttachFieldFlag : public FieldInfoTrait {
 public:
  /*!
   * \brief Attach a field flag to the field
   *
   * \param flag The flag to be set
   *
   * \return The trait object.
   */
  explicit AttachFieldFlag(int32_t flag) : flag_(flag) {}

  /*!
   * \brief Attach kTVMFFIFieldFlagBitMaskSEqHashDef
   */
  TVM_FFI_INLINE static AttachFieldFlag SEqHashDef() {
    return AttachFieldFlag(kTVMFFIFieldFlagBitMaskSEqHashDef);
  }
  /*!
   * \brief Attach kTVMFFIFieldFlagBitMaskSEqHashIgnore
   */
  TVM_FFI_INLINE static AttachFieldFlag SEqHashIgnore() {
    return AttachFieldFlag(kTVMFFIFieldFlagBitMaskSEqHashIgnore);
  }

  TVM_FFI_INLINE void Apply(TVMFFIFieldInfo* info) const { info->flags |= flag_; }

 private:
  int32_t flag_;
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
    if constexpr (std::is_same_v<T, Any>) {
      *reinterpret_cast<T*>(field) = AnyView::CopyFromTVMFFIAny(*value);
    } else {
      *reinterpret_cast<T*>(field) = AnyView::CopyFromTVMFFIAny(*value).cast<T>();
    }
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
  TVM_FFI_INLINE static void ApplyExtraInfoTrait(TVMFFITypeMetadata* info, const T& value) {
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

  template <typename Func>
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
    RegisterFunc(name, GetMethod(std::string(name), std::forward<Func>(func)),
                 std::forward<Extra>(extra)...);
    return *this;
  }

 private:
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
    TVMFFITypeMetadata info;
    info.total_size = sizeof(Class);
    info.structural_eq_hash_kind = Class::_type_s_eq_hash_kind;
    info.creator = nullptr;
    info.doc = TVMFFIByteArray{nullptr, 0};
    if constexpr (std::is_default_constructible_v<Class>) {
      info.creator = ObjectCreatorDefault<Class>;
    }
    // apply extra info traits
    ((ApplyExtraInfoTrait(&info, std::forward<ExtraArgs>(extra_args)), ...));
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterMetadata(type_index_, &info));
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
    Function method = GetMethod(std::string(type_key_) + "." + name, std::forward<Func>(func));
    info.method = AnyView(method).CopyToTVMFFIAny();
    // apply method info traits
    ((ApplyMethodInfoTrait(&info, std::forward<Extra>(extra)), ...));
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterMethod(type_index_, &info));
  }

  int32_t type_index_;
  const char* type_key_;
};

template <typename Class, typename = std::enable_if_t<std::is_base_of_v<Object, Class>>>
class TypeAttrDef : public ReflectionDefBase {
 public:
  template <typename... ExtraArgs>
  explicit TypeAttrDef(ExtraArgs&&... extra_args)
      : type_index_(Class::RuntimeTypeIndex()), type_key_(Class::_type_key) {}

  /*
   * \brief Define a function-valued type attribute.
   *
   * \tparam Func The function type.
   *
   * \param name The name of the function.
   * \param func The function to be registered.
   *
   * \return The TypeAttrDef object.
   */
  template <typename Func>
  TypeAttrDef& def(const char* name, Func&& func) {
    TVMFFIByteArray name_array = {name, std::char_traits<char>::length(name)};
    ffi::Function ffi_func =
        GetMethod(std::string(type_key_) + "." + name, std::forward<Func>(func));
    TVMFFIAny value_any = AnyView(ffi_func).CopyToTVMFFIAny();
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index_, &name_array, &value_any));
    return *this;
  }

  /*
   * \brief Define a constant-valued type attribute.
   *
   * \tparam T The type of the value.
   *
   * \param name The name of the attribute.
   * \param value The value of the attribute.
   *
   * \return The TypeAttrDef object.
   */
  template <typename T>
  TypeAttrDef& attr(const char* name, T value) {
    TVMFFIByteArray name_array = {name, std::char_traits<char>::length(name)};
    TVMFFIAny value_any = AnyView(value).CopyToTVMFFIAny();
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index_, &name_array, &value_any));
    return *this;
  }

 private:
  int32_t type_index_;
  const char* type_key_;
};

/*!
 * \brief Ensure the type attribute column is presented in the system.
 *
 * \param name The name of the type attribute.
 */
inline void EnsureTypeAttrColumn(std::string_view name) {
  TVMFFIByteArray name_array = {name.data(), name.size()};
  AnyView any_view(nullptr);
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(kTVMFFINone, &name_array,
                                                 reinterpret_cast<const TVMFFIAny*>(&any_view)));
}

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_REFLECTION_REGISTRY_H_
