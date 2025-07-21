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
 * \file tvm/node/reflection.h
 * \brief Reflection and serialization of compiler IR/AST nodes.
 */
#ifndef TVM_NODE_REFLECTION_H_
#define TVM_NODE_REFLECTION_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/memory.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include <string>
#include <type_traits>
#include <vector>

namespace tvm {

using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;

/*!
 * \brief Virtual function table to support IR/AST node reflection.
 *
 * Functions are stored in columnar manner.
 * Each column is a vector indexed by Object's type_index.
 */
class ReflectionVTable {
 public:
  /*!
   * \brief Equality comparison function.
   */
  typedef bool (*FSEqualReduce)(const Object* self, const Object* other, SEqualReducer equal);
  /*!
   * \brief Structural hash reduction function.
   */
  typedef void (*FSHashReduce)(const Object* self, SHashReducer hash_reduce);
  /*!
   * \brief creator function.
   * \param repr_bytes Repr bytes to create the object.
   *        If this is not empty then FReprBytes must be defined for the object.
   * \return The created function.
   */
  typedef ObjectPtr<Object> (*FCreate)(const std::string& repr_bytes);
  /*!
   * \brief Function to get a byte representation that can be used to recover the object.
   * \param node The node pointer.
   * \return bytes The bytes that can be used to recover the object.
   */
  typedef std::string (*FReprBytes)(const Object* self);
  /*!
   * \brief Get repr bytes if any.
   * \param self The pointer to the object.
   * \param repr_bytes The output repr bytes, can be null, in which case the function
   *                   simply queries if the ReprBytes function exists for the type.
   * \return Whether repr bytes exists
   */
  inline bool GetReprBytes(const Object* self, std::string* repr_bytes) const;
  /*!
   * \brief Dispatch the SEqualReduce function.
   * \param self The pointer to the object.
   * \param other The pointer to another object to be compared.
   * \param equal The equality comparator.
   * \return the result.
   */
  bool SEqualReduce(const Object* self, const Object* other, SEqualReducer equal) const;
  /*!
   * \brief Dispatch the SHashReduce function.
   * \param self The pointer to the object.
   * \param hash_reduce The hash reducer.
   */
  void SHashReduce(const Object* self, SHashReducer hash_reduce) const;
  /*!
   * \brief Create an initial object using default constructor
   *        by type_key and global key.
   *
   * \param type_key The type key of the object.
   * \param repr_bytes Bytes representation of the object if any.
   */
  TVM_DLL ObjectPtr<Object> CreateInitObject(const std::string& type_key,
                                             const std::string& repr_bytes = "") const;
  /*!
   * \brief Create an object by giving kwargs about its fields.
   *
   * \param type_key The type key.
   * \param kwargs the arguments in format key1, value1, ..., key_n, value_n.
   * \return The created object.
   */
  TVM_DLL ObjectRef CreateObject(const std::string& type_key, const ffi::PackedArgs& kwargs);
  /*!
   * \brief Create an object by giving kwargs about its fields.
   *
   * \param type_key The type key.
   * \param kwargs The field arguments.
   * \return The created object.
   */
  TVM_DLL ObjectRef CreateObject(const std::string& type_key, const Map<String, ffi::Any>& kwargs);
  /*!
   * \brief Get an field object by the attr name.
   * \param self The pointer to the object.
   * \param attr_name The name of the field.
   * \return The corresponding attribute value.
   * \note This function will throw an exception if the object does not contain the field.
   */
  TVM_DLL ffi::Any GetAttr(Object* self, const String& attr_name) const;

  /*!
   * \brief List all the fields in the object.
   * \return All the fields.
   */
  TVM_DLL std::vector<std::string> ListAttrNames(Object* self) const;

  /*! \return The global singleton. */
  TVM_DLL static ReflectionVTable* Global();

  class Registry;
  template <typename T, typename TraitName>
  inline Registry Register();

 private:
  /*! \brief Structural equal function. */
  std::vector<FSEqualReduce> fsequal_reduce_;
  /*! \brief Structural hash function. */
  std::vector<FSHashReduce> fshash_reduce_;
  /*! \brief Creation function. */
  std::vector<FCreate> fcreate_;
  /*! \brief ReprBytes function. */
  std::vector<FReprBytes> frepr_bytes_;
};

/*! \brief Registry of a reflection table. */
class ReflectionVTable::Registry {
 public:
  Registry(ReflectionVTable* parent, uint32_t type_index)
      : parent_(parent), type_index_(type_index) {}
  /*!
   * \brief Set fcreate function.
   * \param f The creator function.
   * \return Reference to self.
   */
  Registry& set_creator(FCreate f) {  // NOLINT(*)
    ICHECK_LT(type_index_, parent_->fcreate_.size());
    parent_->fcreate_[type_index_] = f;
    return *this;
  }
  /*!
   * \brief Set bytes repr function.
   * \param f The ReprBytes function.
   * \return Reference to self.
   */
  Registry& set_repr_bytes(FReprBytes f) {  // NOLINT(*)
    ICHECK_LT(type_index_, parent_->frepr_bytes_.size());
    parent_->frepr_bytes_[type_index_] = f;
    return *this;
  }

 private:
  ReflectionVTable* parent_;
  uint32_t type_index_;
};

#define TVM_REFLECTION_REG_VAR_DEF \
  static TVM_ATTRIBUTE_UNUSED ::tvm::ReflectionVTable::Registry __make_reflection

/*!
 * \brief Directly register reflection VTable.
 * \param TypeName The name of the type.
 * \param TraitName A trait class that implements functions like SEqualReduce.
 *
 * \code
 *
 *  // Example SEQualReduce traits for runtime StringObj.
 *
 *  struct StringObjTrait {
 *
 *
 *    static void SHashReduce(const StringObj* key, SHashReducer hash_reduce) {
 *      hash_reduce->SHashReduceHashedValue(String::StableHashBytes(key->data, key->size));
 *    }
 *
 *    static bool SEqualReduce(const StringObj* lhs,
 *                             const StringObj* rhs,
 *                             SEqualReducer equal) {
 *      if (lhs == rhs) return true;
 *      if (lhs->size != rhs->size) return false;
 *      if (lhs->data != rhs->data) return true;
 *      return std::memcmp(lhs->data, rhs->data, lhs->size) != 0;
 *    }
 *  };
 *
 *  TVM_REGISTER_REFLECTION_VTABLE(StringObj, StringObjTrait);
 *
 * \endcode
 *
 * \note This macro can be called in different place as TVM_REGISTER_OBJECT_TYPE.
 *       And can be used to register the related reflection functions for runtime objects.
 */
#define TVM_REGISTER_REFLECTION_VTABLE(TypeName, TraitName) \
  TVM_STR_CONCAT(TVM_REFLECTION_REG_VAR_DEF, __COUNTER__) = \
      ::tvm::ReflectionVTable::Global()->Register<TypeName, TraitName>()

/*!
 * \brief Register a node type to object registry and reflection registry.
 * \param TypeName The name of the type.
 * \note This macro will call TVM_REGISTER_OBJECT_TYPE for the type as well.
 */
#define TVM_REGISTER_NODE_TYPE(TypeName)                                             \
  TVM_REGISTER_OBJECT_TYPE(TypeName);                                                \
  TVM_REGISTER_REFLECTION_VTABLE(TypeName, ::tvm::detail::ReflectionTrait<TypeName>) \
      .set_creator([](const std::string&) -> ObjectPtr<Object> {                     \
        return ::tvm::ffi::make_object<TypeName>();                                  \
      })

// Implementation details
namespace detail {

template <typename T, bool = T::_type_has_method_sequal_reduce>
struct ImplSEqualReduce {
  static constexpr const std::nullptr_t SEqualReduce = nullptr;
};

template <typename T>
struct ImplSEqualReduce<T, true> {
  static bool SEqualReduce(const T* self, const T* other, SEqualReducer equal) {
    return self->SEqualReduce(other, equal);
  }
};

template <typename T, bool = T::_type_has_method_shash_reduce>
struct ImplSHashReduce {
  static constexpr const std::nullptr_t SHashReduce = nullptr;
};

template <typename T>
struct ImplSHashReduce<T, true> {
  static void SHashReduce(const T* self, SHashReducer hash_reduce) {
    self->SHashReduce(hash_reduce);
  }
};

template <typename T>
struct ReflectionTrait : public ImplSEqualReduce<T>, public ImplSHashReduce<T> {};

template <typename T, typename TraitName,
          bool = std::is_null_pointer<decltype(TraitName::SEqualReduce)>::value>
struct SelectSEqualReduce {
  static constexpr const std::nullptr_t SEqualReduce = nullptr;
};

template <typename T, typename TraitName>
struct SelectSEqualReduce<T, TraitName, false> {
  static bool SEqualReduce(const Object* self, const Object* other, SEqualReducer equal) {
    return TraitName::SEqualReduce(static_cast<const T*>(self), static_cast<const T*>(other),
                                   equal);
  }
};

template <typename T, typename TraitName,
          bool = std::is_null_pointer<decltype(TraitName::SHashReduce)>::value>
struct SelectSHashReduce {
  static constexpr const std::nullptr_t SHashReduce = nullptr;
};

template <typename T, typename TraitName>
struct SelectSHashReduce<T, TraitName, false> {
  static void SHashReduce(const Object* self, SHashReducer hash_reduce) {
    return TraitName::SHashReduce(static_cast<const T*>(self), hash_reduce);
  }
};

}  // namespace detail

template <typename T, typename TraitName>
inline ReflectionVTable::Registry ReflectionVTable::Register() {
  uint32_t tindex = T::RuntimeTypeIndex();
  if (tindex >= fcreate_.size()) {
    fcreate_.resize(tindex + 1, nullptr);
    frepr_bytes_.resize(tindex + 1, nullptr);
    fsequal_reduce_.resize(tindex + 1, nullptr);
    fshash_reduce_.resize(tindex + 1, nullptr);
  }
  // functor that implements the redirection.
  fsequal_reduce_[tindex] = ::tvm::detail::SelectSEqualReduce<T, TraitName>::SEqualReduce;

  fshash_reduce_[tindex] = ::tvm::detail::SelectSHashReduce<T, TraitName>::SHashReduce;

  return Registry(this, tindex);
}

inline bool ReflectionVTable::GetReprBytes(const Object* self, std::string* repr_bytes) const {
  uint32_t tindex = self->type_index();
  if (tindex < frepr_bytes_.size() && frepr_bytes_[tindex] != nullptr) {
    if (repr_bytes != nullptr) {
      *repr_bytes = frepr_bytes_[tindex](self);
    }
    return true;
  } else {
    return false;
  }
}

/*!
 * \brief Given an object and an address of its attribute, return the key of the attribute.
 * \return nullptr if no attribute with the given address exists.
 */
Optional<String> GetAttrKeyByAddress(const Object* object, const void* attr_address);

}  // namespace tvm
#endif  // TVM_NODE_REFLECTION_H_
