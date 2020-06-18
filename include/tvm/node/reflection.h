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

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/data_type.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>

#include <vector>
#include <string>
#include <type_traits>

namespace tvm {

using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;

/*!
 * \brief Visitor class for to get the attributesof a AST/IR node.
 *  The content is going to be called for each field.
 *
 *  Each objects that wants reflection will need to implement
 *  a VisitAttrs function and call visitor->Visit on each of its field.
 */
class AttrVisitor {
 public:
//! \cond Doxygen_Suppress
  TVM_DLL virtual ~AttrVisitor() = default;
  TVM_DLL virtual void Visit(const char* key, double* value) = 0;
  TVM_DLL virtual void Visit(const char* key, int64_t* value) = 0;
  TVM_DLL virtual void Visit(const char* key, uint64_t* value) = 0;
  TVM_DLL virtual void Visit(const char* key, int* value) = 0;
  TVM_DLL virtual void Visit(const char* key, bool* value) = 0;
  TVM_DLL virtual void Visit(const char* key, std::string* value) = 0;
  TVM_DLL virtual void Visit(const char* key, void** value) = 0;
  TVM_DLL virtual void Visit(const char* key, DataType* value) = 0;
  TVM_DLL virtual void Visit(const char* key, runtime::NDArray* value) = 0;
  TVM_DLL virtual void Visit(const char* key, runtime::ObjectRef* value) = 0;
  template<typename ENum,
           typename = typename std::enable_if<std::is_enum<ENum>::value>::type>
  void Visit(const char* key, ENum* ptr) {
    static_assert(std::is_same<int, typename std::underlying_type<ENum>::type>::value,
                  "declare enum to be enum int to use visitor");
    this->Visit(key, reinterpret_cast<int*>(ptr));
  }
//! \endcond
};

/*!
 * \brief Virtual function table to support IR/AST node reflection.
 *
 * Functions are stored  in columar manner.
 * Each column is a vector indexed by Object's type_index.
 */
class ReflectionVTable {
 public:
  /*!
   * \brief Visitor function.
   * \note We use function pointer, instead of std::function
   *       to reduce the dispatch overhead as field visit
   *       does not need as much customization.
   */
  typedef void (*FVisitAttrs)(Object* self, AttrVisitor* visitor);
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
   * \brief Dispatch the VisitAttrs function.
   * \param self The pointer to the object.
   * \param visitor The attribute visitor.
   */
  inline void VisitAttrs(Object* self, AttrVisitor* visitor) const;
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
   * \return the result.
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
   * \brief Get an field object by the attr name.
   * \param self The pointer to the object.
   * \param attr_name The name of the field.
   * \return The corresponding attribute value.
   * \note This function will throw an exception if the object does not contain the field.
   */
  TVM_DLL runtime::TVMRetValue GetAttr(Object* self, const std::string& attr_name) const;

  /*!
   * \brief List all the fields in the object.
   * \return All the fields.
   */
  TVM_DLL std::vector<std::string> ListAttrNames(Object* self) const;

  /*! \return The global singleton. */
  TVM_DLL static ReflectionVTable* Global();

  class Registry;
  template<typename T, typename TraitName>
  inline Registry Register();

 private:
  /*! \brief Attribute visitor. */
  std::vector<FVisitAttrs> fvisit_attrs_;
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
      : parent_(parent), type_index_(type_index) { }
  /*!
   * \brief Set fcreate function.
   * \param f The creator function.
   * \return rference to self.
   */
  Registry& set_creator(FCreate f) {  // NOLINT(*)
    CHECK_LT(type_index_, parent_->fcreate_.size());
    parent_->fcreate_[type_index_] = f;
    return *this;
  }
  /*!
   * \brief Set bytes repr function.
   * \param f The ReprBytes function.
   * \return rference to self.
   */
  Registry& set_repr_bytes(FReprBytes f) {  // NOLINT(*)
    CHECK_LT(type_index_, parent_->frepr_bytes_.size());
    parent_->frepr_bytes_[type_index_] = f;
    return *this;
  }

 private:
  ReflectionVTable* parent_;
  uint32_t type_index_;
};


#define TVM_REFLECTION_REG_VAR_DEF                                     \
  static TVM_ATTRIBUTE_UNUSED ::tvm::ReflectionVTable::Registry        \
  __make_reflectiion

/*!
 * \brief Directly register reflection VTable.
 * \param TypeName The name of the type.
 * \param TraitName A trait class that implements functions like VisitAttrs and SEqualReduce.
 *
 * \code
 *
 *  // Example SEQualReduce traits for runtime StringObj.
 *
 *  struct StringObjTrait {
 *     static constexpr const std::nullptr_t VisitAttrs = nullptr;
 *
 *    static bool SEqualReduce(const runtime::StringObj* lhs,
 *                             const runtime::StringObj* rhs,
 *                             SEqualReducer equal) {
 *      if (lhs == rhs) return true;
 *      if (lhs->size != rhs->size) return false;
 *      if (lhs->data != rhs->data) return true;
 *      return std::memcmp(lhs->data, rhs->data, lhs->size) != 0;
 *    }
 *  };
 *
 *  TVM_REGISTER_REFLECTION_VTABLE(runtime::StringObj, StringObjTrait);
 *
 * \endcode
 *
 * \note This macro can be called in different place as TVM_REGISTER_OBJECT_TYPE.
 *       And can be used to register the related reflection functions for runtime objects.
 */
#define TVM_REGISTER_REFLECTION_VTABLE(TypeName, TraitName)             \
  TVM_STR_CONCAT(TVM_REFLECTION_REG_VAR_DEF, __COUNTER__) =             \
      ::tvm::ReflectionVTable::Global()->Register<TypeName, TraitName>() \

/*!
 * \brief Register a node type to object registry and reflection registry.
 * \param TypeName The name of the type.
 * \note This macro will call TVM_REGISTER_OBJECT_TYPE for the type as well.
 */
#define TVM_REGISTER_NODE_TYPE(TypeName)                                \
  TVM_REGISTER_OBJECT_TYPE(TypeName);                                   \
  TVM_REGISTER_REFLECTION_VTABLE(TypeName, ::tvm::detail::ReflectionTrait<TypeName>) \
  .set_creator([](const std::string&) -> ObjectPtr<Object> {            \
      return ::tvm::runtime::make_object<TypeName>();                   \
    })


// Implementation details
namespace detail {

template<typename T,
         bool = T::_type_has_method_visit_attrs>
struct ImplVisitAttrs {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;
};

template<typename T>
struct ImplVisitAttrs<T, true> {
  static void VisitAttrs(T* self, AttrVisitor* v) {
    self->VisitAttrs(v);
  }
};

template<typename T,
         bool = T::_type_has_method_sequal_reduce>
struct ImplSEqualReduce {
  static constexpr const std::nullptr_t SEqualReduce = nullptr;
};

template<typename T>
struct ImplSEqualReduce<T, true> {
  static bool SEqualReduce(const T* self, const T* other, SEqualReducer equal) {
    return self->SEqualReduce(other, equal);
  }
};

template<typename T,
         bool = T::_type_has_method_shash_reduce>
struct ImplSHashReduce {
  static constexpr const std::nullptr_t SHashReduce = nullptr;
};

template<typename T>
struct ImplSHashReduce<T, true> {
  static void SHashReduce(const T* self, SHashReducer hash_reduce) {
    self->SHashReduce(hash_reduce);
  }
};

template<typename T>
struct ReflectionTrait :
      public ImplVisitAttrs<T>,
      public ImplSEqualReduce<T>,
      public ImplSHashReduce<T> {
};

template<typename T, typename TraitName,
         bool = std::is_null_pointer<decltype(TraitName::VisitAttrs)>::value>
struct SelectVisitAttrs {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;
};

template<typename T, typename TraitName>
struct SelectVisitAttrs<T, TraitName, false> {
  static void VisitAttrs(Object* self, AttrVisitor* v) {
    TraitName::VisitAttrs(static_cast<T*>(self), v);
  }
};

template<typename T, typename TraitName,
         bool = std::is_null_pointer<decltype(TraitName::SEqualReduce)>::value>
struct SelectSEqualReduce {
  static constexpr const std::nullptr_t SEqualReduce = nullptr;
};

template<typename T, typename TraitName>
struct SelectSEqualReduce<T, TraitName, false> {
  static bool SEqualReduce(const Object* self,
                           const Object* other,
                           SEqualReducer equal) {
    return TraitName::SEqualReduce(static_cast<const T*>(self),
                                   static_cast<const T*>(other),
                                   equal);
  }
};

template<typename T, typename TraitName,
         bool = std::is_null_pointer<decltype(TraitName::SHashReduce)>::value>
struct SelectSHashReduce {
  static constexpr const std::nullptr_t SHashReduce = nullptr;
};

template<typename T, typename TraitName>
struct SelectSHashReduce<T, TraitName, false> {
  static void SHashReduce(const Object* self,
                          SHashReducer hash_reduce) {
    return TraitName::SHashReduce(static_cast<const T*>(self),
                                  hash_reduce);
  }
};

}  // namespace detail

template<typename T, typename TraitName>
inline ReflectionVTable::Registry
ReflectionVTable::Register() {
  uint32_t tindex = T::RuntimeTypeIndex();
  if (tindex >= fvisit_attrs_.size()) {
    fvisit_attrs_.resize(tindex + 1, nullptr);
    fcreate_.resize(tindex + 1, nullptr);
    frepr_bytes_.resize(tindex + 1, nullptr);
    fsequal_reduce_.resize(tindex + 1, nullptr);
    fshash_reduce_.resize(tindex + 1, nullptr);
  }
  // functor that implemnts the redirection.
  fvisit_attrs_[tindex] =
      ::tvm::detail::SelectVisitAttrs<T, TraitName>::VisitAttrs;

  fsequal_reduce_[tindex] =
      ::tvm::detail::SelectSEqualReduce<T, TraitName>::SEqualReduce;

  fshash_reduce_[tindex] =
      ::tvm::detail::SelectSHashReduce<T, TraitName>::SHashReduce;

  return Registry(this, tindex);
}

inline void ReflectionVTable::
VisitAttrs(Object* self, AttrVisitor* visitor) const {
  uint32_t tindex = self->type_index();
  if (tindex >= fvisit_attrs_.size() || fvisit_attrs_[tindex] == nullptr) {
    LOG(FATAL) << "TypeError: " << self->GetTypeKey()
               << " is not registered via TVM_REGISTER_NODE_TYPE";
  }
  fvisit_attrs_[tindex](self, visitor);
}

inline bool ReflectionVTable::GetReprBytes(const Object* self,
                                           std::string* repr_bytes) const {
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

}  // namespace tvm
#endif  // TVM_NODE_REFLECTION_H_
