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
 * \file tvm/ffi/reflection/creator.h
 * \brief Reflection-based creator to create objects from type key and fields.
 */
#ifndef TVM_FFI_REFLECTION_CREATOR_H_
#define TVM_FFI_REFLECTION_CREATOR_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/string.h>

namespace tvm {
namespace ffi {
namespace reflection {
/*!
 * \brief helper wrapper class of TVMFFITypeInfo to create object based on reflection.
 */
class ObjectCreator {
 public:
  /*!
   * \brief Constructor
   * \param type_key The type key.
   */
  explicit ObjectCreator(std::string_view type_key)
      : ObjectCreator(TVMFFIGetTypeInfo(TypeKeyToIndex(type_key))) {}

  /*!
   * \brief Constructor
   * \param type_info The type info.
   */
  explicit ObjectCreator(const TVMFFITypeInfo* type_info) : type_info_(type_info) {
    int32_t type_index = type_info->type_index;
    if (type_info->metadata == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "Type `" << TypeIndexToTypeKey(type_index)
                                  << "` does not have reflection registered";
    }
    if (type_info->metadata->creator == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "Type `" << TypeIndexToTypeKey(type_index)
                                  << "` does not support default constructor, "
                                  << "as a result cannot be created via reflection";
    }
  }

  /**
   * \brief Create an object from a map of fields.
   * \param fields The fields of the object.
   * \return The created object.
   */
  Any operator()(const Map<String, Any>& fields) const {
    TVMFFIObjectHandle handle;
    TVM_FFI_CHECK_SAFE_CALL(type_info_->metadata->creator(&handle));
    ObjectPtr<Object> ptr =
        details::ObjectUnsafe::ObjectPtrFromOwned<Object>(static_cast<TVMFFIObject*>(handle));
    size_t match_field_count = 0;
    ForEachFieldInfo(type_info_, [&](const TVMFFIFieldInfo* field_info) {
      String field_name(field_info->name);
      void* field_addr = reinterpret_cast<char*>(ptr.get()) + field_info->offset;
      if (fields.count(field_name) != 0) {
        Any field_value = fields[field_name];
        field_info->setter(field_addr, reinterpret_cast<const TVMFFIAny*>(&field_value));
        ++match_field_count;
      } else if (field_info->flags & kTVMFFIFieldFlagBitMaskHasDefault) {
        field_info->setter(field_addr, &(field_info->default_value));
      } else {
        TVM_FFI_THROW(TypeError) << "Required field `"
                                 << String(field_info->name.data, field_info->name.size)
                                 << "` not set in type `"
                                 << String(type_info_->type_key.data, type_info_->type_key.size)
                                 << "`";
      }
    });
    if (match_field_count == fields.size()) return ObjectRef(ptr);
    // report error that checks if contains extra fields that are not in the type
    auto check_field_name = [&](const String& field_name) {
      bool found = false;
      ForEachFieldInfoWithEarlyStop(type_info_, [&](const TVMFFIFieldInfo* field_info) {
        if (field_name.compare(field_info->name) == 0) {
          found = true;
          return true;
        }
        return false;
      });
      return found;
    };
    for (const auto& [field_name, _] : fields) {
      if (!check_field_name(field_name)) {
        TVM_FFI_THROW(TypeError) << "Type `"
                                 << String(type_info_->type_key.data, type_info_->type_key.size)
                                 << "` does not have field `" << field_name << "`";
      }
    }
    TVM_FFI_UNREACHABLE();
  }

 private:
  const TVMFFITypeInfo* type_info_;
};
}  // namespace reflection
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_REFLECTION_CREATOR_H_
