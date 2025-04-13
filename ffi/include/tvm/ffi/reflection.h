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
 * \file tvm/ffi/reflection.h
 * \brief Base reflection support to access object fields.
 */
#ifndef TVM_FFI_REFLECTION_H_
#define TVM_FFI_REFLECTION_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/type_traits.h>

namespace tvm {
namespace ffi {
namespace details {

template <typename T, typename = void>
struct TypeToFieldStaticTypeIndex {
  static constexpr int32_t value = TypeIndex::kTVMFFIAny;
};

template <typename T>
struct TypeToFieldStaticTypeIndex<T, std::enable_if_t<TypeTraits<T>::convert_enabled>> {
  static constexpr int32_t value = TypeTraits<T>::field_static_type_index;
};

template <typename T, typename = void>
struct TypeToRuntimeTypeIndex {
  static int32_t v() { return TypeToFieldStaticTypeIndex<T>::value; }
};

template <typename T>
struct TypeToRuntimeTypeIndex<T, std::enable_if_t<std::is_base_of_v<ObjectRef, T>>> {
  static int32_t v() { return T::ContainerType::RuntimeTypeIndex(); }
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
inline int64_t GetFieldByteOffsetToObject(T Class::*field_ptr) {
  int64_t field_offset_to_class =
      reinterpret_cast<int64_t>(&(static_cast<Class*>(nullptr)->*field_ptr));
  return field_offset_to_class - details::ObjectUnsafe::GetObjectOffsetToSubclass<Class>();
}

class ReflectionDef {
 public:
  explicit ReflectionDef(int32_t type_index) : type_index_(type_index) {}

  template <typename Class, typename T>
  ReflectionDef& def_readonly(const char* name, T Class::*field_ptr) {
    RegisterField(name, field_ptr, true);
    return *this;
  }

  template <typename Class, typename T>
  ReflectionDef& def_readwrite(const char* name, T Class::*field_ptr) {
    RegisterField(name, field_ptr, false);
    return *this;
  }

  operator int32_t() const { return type_index_; }

 private:
  template <typename Class, typename T>
  void RegisterField(const char* name, T Class::*field_ptr, bool readonly) {
    TVMFFIFieldInfo info;
    info.name = name;
    info.field_static_type_index = TypeToFieldStaticTypeIndex<T>::value;
    // store byte offset and setter, getter
    // so the same setter can be reused for all the same type
    info.byte_offset = GetFieldByteOffsetToObject<Class, T>(field_ptr);
    info.readonly = readonly;
    info.getter = FieldGetter<T>;
    info.setter = FieldSetter<T>;
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIRegisterTypeField(type_index_, &info));
  }

  template <typename T>
  static int FieldGetter(void* field, TVMFFIAny* result) {
    TVM_FFI_SAFE_CALL_BEGIN();
    *result = details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(*reinterpret_cast<T*>(field)));
    TVM_FFI_SAFE_CALL_END();
  }

  template <typename T>
  static int FieldSetter(void* field, const TVMFFIAny* value) {
    TVM_FFI_SAFE_CALL_BEGIN();
    *reinterpret_cast<T*>(field) = AnyView::CopyFromTVMFFIAny(*value);
    TVM_FFI_SAFE_CALL_END();
  }

  int32_t type_index_;
};

/*!
 * \brief helper function to get reflection field info by type key and field name
 */
inline const TVMFFIFieldInfo* GetReflectionFieldInfo(const char* type_key, const char* field_name) {
  int32_t type_index;
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKey2Index(type_key, &type_index));
  const TypeInfo* info = TVMFFIGetTypeInfo(type_index);
  for (int32_t i = 0; i < info->num_fields; ++i) {
    if (std::strcmp(info->fields[i].name, field_name) == 0) {
      return &(info->fields[i]);
    }
  }
  TVM_FFI_THROW(RuntimeError) << "Cannot find field " << field_name << " in " << type_key;
}

/*!
 * \brief helper wrapper class to obtain a getter.
 */
class ReflectionFieldGetter {
 public:
  explicit ReflectionFieldGetter(const TVMFFIFieldInfo* field_info) : field_info_(field_info) {}

  Any operator()(const Object* obj_ptr) const {
    Any result;
    const void* addr = reinterpret_cast<const char*>(obj_ptr) + field_info_->byte_offset;
    TVM_FFI_CHECK_SAFE_CALL(
        field_info_->getter(const_cast<void*>(addr), reinterpret_cast<TVMFFIAny*>(&result)));
    return result;
  }

  Any operator()(const ObjectPtr<Object>& obj_ptr) const { return operator()(obj_ptr.get()); }

  Any operator()(const ObjectRef& obj) const { return operator()(obj.get()); }

 private:
  const TVMFFIFieldInfo* field_info_;
};

}  // namespace details

/*!
 * \brief helper function to get type index from key
 */
inline int32_t TypeKey2Index(const char* type_key) {
  int32_t type_index;
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKey2Index(type_key, &type_index));
  return type_index;
}

/*!
 * \brief helper macro to declare a base object type that can be inherited.
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 */
#define TVM_FFI_DECLARE_BASE_OBJECT_INFO(TypeName, ParentType)                                \
  static constexpr int32_t _type_depth = ParentType::_type_depth + 1;                         \
  static int32_t _GetOrAllocRuntimeTypeIndex() {                                              \
    static_assert(!ParentType::_type_final, "ParentType marked as final");                    \
    static_assert(TypeName::_type_child_slots == 0 || ParentType::_type_child_slots == 0 ||   \
                      TypeName::_type_child_slots < ParentType::_type_child_slots,            \
                  "Need to set _type_child_slots when parent specifies it.");                 \
    static int32_t tindex = TVMFFIGetOrAllocTypeIndex(                                        \
        TypeName::_type_key, -1, TypeName::_type_depth, TypeName::_type_child_slots,          \
        TypeName::_type_child_slots_can_overflow, ParentType::_GetOrAllocRuntimeTypeIndex()); \
    return tindex;                                                                            \
  }                                                                                           \
  static int32_t RuntimeTypeIndex() { return _GetOrAllocRuntimeTypeIndex(); }                 \
  static inline int32_t _type_index =                                                         \
      ::tvm::ffi::details::ReflectionDef(_GetOrAllocRuntimeTypeIndex())

/*!
 * \brief helper macro to declare type information in a final class.
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 */
#define TVM_FFI_DECLARE_FINAL_OBJECT_INFO(TypeName, ParentType) \
  static const constexpr int _type_child_slots = 0;             \
  static const constexpr bool _type_final = true;               \
  TVM_FFI_DECLARE_BASE_OBJECT_INFO(TypeName, ParentType)

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_REFLECTION_H_
