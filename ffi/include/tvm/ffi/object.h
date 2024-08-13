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
 * \file tvm/ffi/object.h
 * \brief A managed object in the TVM FFI.
 */
#ifndef TVM_FFI_OBJECT_H_
#define TVM_FFI_OBJECT_H_

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/internal_utils.h>

#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

using TypeIndex = TVMFFITypeIndex;

namespace details {
// forward declare object internal
struct ObjectInternal;
}  // namespace details

/*!
 * \brief base class of all object containers.
 *
 * Sub-class of objects should declare the following static constexpr fields:
 *
 * - _type_index:
 *      Static type index of the object, if assigned to TypeIndex::kTVMFFIDynObject
 *      the type index will be assigned during runtime.
 *      Runtime type index can be accessed by ObjectType::TypeIndex();
 * - _type_key:
 *       The unique string identifier of the type.
 * - _type_final:
 *       Whether the type is terminal type(there is no subclass of the type in the object system).
 *       This field is automatically set by macro TVM_DECLARE_FINAL_OBJECT_INFO
 *       It is still OK to sub-class a terminal object type T and construct it using make_object.
 *       But IsInstance check will only show that the object type is T(instead of the sub-class).
 *
 * The following two fields are necessary for base classes that can be sub-classed.
 *
 * - _type_child_slots:
 *       Number of reserved type index slots for child classes.
 *       Used for runtime optimization for type checking in IsInstance.
 *       If an object's type_index is within range of [type_index, type_index + _type_child_slots]
 *       Then the object can be quickly decided as sub-class of the current object class.
 *       If not, a fallback mechanism is used to check the global type table.
 *       Recommendation: set to estimate number of children needed.
 *
 * - _type_child_slots_can_overflow:
 *       Whether we can add additional child classes even if the number of child classes
 *       exceeds the _type_child_slots. A fallback mechanism to check type table will be used.
 *       Recommendation: set to false for optimal runtime speed if we know exact number of children.
 *
 * Two macros are used to declare helper functions in the object:
 * - Use TVM_FFI_DECLARE_BASE_OBJECT_INFO for object classes that can be sub-classed.
 * - Use TVM_FFI_DECLARE_FINAL_OBJECT_INFO for object classes that cannot be sub-classed.
 *
 * New objects can be created using make_object function.
 * Which will automatically populate the type_index and deleter of the object.
 */
class Object {
 private:
  /*! \brief header field that is the common prefix of all objects */
  TVMFFIObject header_;

 public:
  Object() {
    header_.ref_counter = 0;
    header_.deleter = nullptr;
  }

  // Information about the object
  static constexpr const char* _type_key = "runtime.Object";

  // Default object type properties for sub-classes
  static constexpr bool _type_final = false;
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_child_slots_can_overflow = true;
  // NOTE: the following field is not type index of Object
  // but was intended to be used by sub-classes as default value.
  // The type index of Object is TypeIndex::kRoot
  static constexpr int32_t _type_index = TypeIndex::kTVMFFIObject;

  // The following functions are provided by macro
  // TVM_FFI_DECLARE_BASE_OBJECT_INFO and TVM_DECLARE_FINAL_OBJECT_INFO
  /*!
   * \brief Get the runtime allocated type index of the type
   * \note Getting this information may need dynamic calls into a global table.
   */
  static int32_t RuntimeTypeIndex() { return TypeIndex::kTVMFFIObject; }
  /*!
   * \brief Internal function to get or allocate a runtime index.
   * \note
   */
  static int32_t _GetOrAllocRuntimeTypeIndex() { return TypeIndex::kTVMFFIObject; }

 private:
  /*! \brief increase reference count */
  void IncRef() { details::AtomicIncrementRelaxed(&(header_.ref_counter)); }

  /*! \brief decrease reference count and delete the object */
  void DecRef() {
    if (details::AtomicDecrementRelAcq(&(header_.ref_counter)) == 1) {
      if (header_.deleter != nullptr) {
        header_.deleter(this);
      }
    }
  }

  /*!
   * \return The usage count of the cell.
   * \note We use stl style naming to be consistent with known API in shared_ptr.
   */
  int32_t use_count() const { return details::AtomicLoadRelaxed(&(header_.ref_counter)); }

  // friend classes
  template <typename>
  friend class ObjectPtr;
  friend class tvm::ffi::details::ObjectInternal;
};

/*!
 * \brief A custom smart pointer for Object.
 * \tparam T the content data type.
 * \sa make_object
 */
template <typename T>
class ObjectPtr {
 public:
  /*! \brief default constructor */
  ObjectPtr() {}
  /*! \brief default constructor */
  ObjectPtr(std::nullptr_t) {}  // NOLINT(*)
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  ObjectPtr(const ObjectPtr<T>& other)  // NOLINT(*)
      : ObjectPtr(other.data_) {}
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  template <typename U>
  ObjectPtr(const ObjectPtr<U>& other)  // NOLINT(*)
      : ObjectPtr(other.data_) {
    static_assert(std::is_base_of<T, U>::value,
                  "can only assign of child class ObjectPtr to parent");
  }
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  ObjectPtr(ObjectPtr<T>&& other)  // NOLINT(*)
      : data_(other.data_) {
    other.data_ = nullptr;
  }
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  template <typename Y>
  ObjectPtr(ObjectPtr<Y>&& other)  // NOLINT(*)
      : data_(other.data_) {
    static_assert(std::is_base_of<T, Y>::value,
                  "can only assign of child class ObjectPtr to parent");
    other.data_ = nullptr;
  }
  /*! \brief destructor */
  ~ObjectPtr() { this->reset(); }
  /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
  void swap(ObjectPtr<T>& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }
  /*!
   * \return Get the content of the pointer
   */
  T* get() const { return static_cast<T*>(data_); }
  /*!
   * \return The pointer
   */
  T* operator->() const { return get(); }
  /*!
   * \return The reference
   */
  T& operator*() const {  // NOLINT(*)
    return *get();
  }
  /*!
   * \brief copy assignment
   * \param other The value to be assigned.
   * \return reference to self.
   */
  ObjectPtr<T>& operator=(const ObjectPtr<T>& other) {  // NOLINT(*)
    // takes in plane operator to enable copy elison.
    // copy-and-swap idiom
    ObjectPtr(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*!
   * \brief move assignment
   * \param other The value to be assigned.
   * \return reference to self.
   */
  ObjectPtr<T>& operator=(ObjectPtr<T>&& other) {  // NOLINT(*)
    // copy-and-swap idiom
    ObjectPtr(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*!
   * \brief nullptr check
   * \return result of comparison of internal pointer with nullptr.
   */
  explicit operator bool() const { return get() != nullptr; }
  /*! \brief reset the content of ptr to be nullptr */
  void reset() {
    if (data_ != nullptr) {
      data_->DecRef();
      data_ = nullptr;
    }
  }
  /*! \return The use count of the ptr, for debug purposes */
  int use_count() const { return data_ != nullptr ? data_->use_count() : 0; }
  /*! \return whether the reference is unique */
  bool unique() const { return data_ != nullptr && data_->use_count() == 1; }
  /*! \return Whether two ObjectPtr do not equal each other */
  bool operator==(const ObjectPtr<T>& other) const { return data_ == other.data_; }
  /*! \return Whether two ObjectPtr equals each other */
  bool operator!=(const ObjectPtr<T>& other) const { return data_ != other.data_; }
  /*! \return Whether the pointer is nullptr */
  bool operator==(std::nullptr_t) const { return data_ == nullptr; }
  /*! \return Whether the pointer is not nullptr */
  bool operator!=(std::nullptr_t) const { return data_ != nullptr; }

 private:
  /*! \brief internal pointer field */
  Object* data_{nullptr};
  /*!
   * \brief constructor from Object
   * \param data The data pointer
   */
  explicit ObjectPtr(Object* data) : data_(data) {
    if (data != nullptr) {
      data_->IncRef();
    }
  }
  // friend classes
  friend class Object;
  friend class ObjectRef;
  friend struct ObjectPtrHash;
  template <typename>
  friend class ObjectPtr;
  friend class tvm::ffi::details::ObjectInternal;
  template <typename RelayRefType, typename ObjType>
  friend RelayRefType GetRef(const ObjType* ptr);
  template <typename BaseType, typename ObjType>
  friend ObjectPtr<BaseType> GetObjectPtr(ObjType* ptr);
};


// Forward declaration, to prevent circular includes.
template <typename T>
class Optional;

/*! \brief Base class of all object reference */
class ObjectRef {
 public:
  /*! \brief default constructor */
  ObjectRef() = default;
  /*! \brief Constructor from existing object ptr */
  explicit ObjectRef(ObjectPtr<Object> data) : data_(data) {}
  /*!
   * \brief Comparator
   * \param other Another object ref.
   * \return the compare result.
   */
  bool same_as(const ObjectRef& other) const { return data_ == other.data_; }
  /*!
   * \brief Comparator
   * \param other Another object ref.
   * \return the compare result.
   */
  bool operator==(const ObjectRef& other) const { return data_ == other.data_; }
  /*!
   * \brief Comparator
   * \param other Another object ref.
   * \return the compare result.
   */
  bool operator!=(const ObjectRef& other) const { return data_ != other.data_; }
  /*!
   * \brief Comparator
   * \param other Another object ref by address.
   * \return the compare result.
   */
  bool operator<(const ObjectRef& other) const { return data_.get() < other.data_.get(); }
  /*!
   * \return whether the object is defined(not null).
   */
  bool defined() const { return data_ != nullptr; }
  /*! \return the internal object pointer */
  const Object* get() const { return data_.get(); }
  /*! \return the internal object pointer */
  const Object* operator->() const { return get(); }
  /*! \return whether the reference is unique */
  bool unique() const { return data_.unique(); }
  /*! \return The use count of the ptr, for debug purposes */
  int use_count() const { return data_.use_count(); }

  /*!
   * \brief Try to downcast the internal Object to a
   *  raw pointer of a corresponding type.
   *
   *  The function will return a nullptr if the cast failed.
   *
   *      if (const AddNode *ptr = node_ref.as<AddNode>()) {
   *        // This is an add node
   *      }
   *
   * \tparam ObjectType the target type, must be a subtype of Object
   */
  template <typename ObjectType, typename = std::enable_if_t<std::is_base_of_v<Object, ObjectType>>>
  inline const ObjectType* as() const;

  /*! \brief type indicate the container type. */
  using ContainerType = Object;
  // Default type properties for the reference class.
  static constexpr bool _type_is_nullable = true;

 protected:
  /*! \brief Internal pointer that backs the reference. */
  ObjectPtr<Object> data_;
  /*! \return return a mutable internal ptr, can be used by sub-classes. */
  Object* get_mutable() const { return data_.get(); }
  // friend classes.
  friend struct ObjectPtrHash;
  friend class tvm::ffi::details::ObjectInternal;
  template <typename SubRef, typename BaseRef>
  friend SubRef Downcast(BaseRef ref);
};

/*!
 * \brief Get an object ptr type from a raw object ptr.
 *
 * \param ptr The object pointer
 * \tparam BaseType The reference type
 * \tparam ObjectType The object type
 * \return The corresponding RefType
 */
template <typename BaseType, typename ObjectType>
inline ObjectPtr<BaseType> GetObjectPtr(ObjectType* ptr);

/*!
 * \brief Helper macro to declare list of static checks about object meta-data.
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 */
#define TVM_FFI_OBJECT_STATIC_CHECKS(TypeName, ParentType)                                \
  static_assert(!ParentType::_type_final, "ParentType marked as final");                  \
  static_assert(TypeName::_type_child_slots == 0 || ParentType::_type_child_slots == 0 || \
                    TypeName::_type_child_slots < ParentType::_type_child_slots,          \
                "Need to set _type_child_slots when parent specifies it.");               \
  static_assert(TypeName::_type_child_slots == 0 || ParentType::_type_child_slots == 0 || \
                    TypeName::_type_child_slots < ParentType::_type_child_slots,          \
                "Need to set _type_child_slots when parent specifies it.");

/*!
 * \brief Helper macro to declare a object that comes with static type index.
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 */
#define TVM_FFI_DECLARE_STATIC_OBJECT_INFO(TypeName, ParentType) \
  TVM_FFI_OBJECT_STATIC_CHECKS(TypeName, ParentType)             \
  static int32_t RuntimeTypeIndex() { return TypeName::_type_index; }

/*!
 * \brief helper macro to declare a base object type that can be inherited.
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 */
#define TVM_DECLARE_BASE_OBJECT_INFO(TypeName, ParentType)                      \
  static_assert(TVM_FFI_ALLOW_DYN_TYPE,                                         \
                "Dynamic object depend on TVM_FFI_ALLOW_DYN_TYPE cd set to 1"); \
  TVM_FFI_OBJECT_STATIC_CHECKS(TypaName, ParentType)                            \
  static inline int32_t _type_index = _GetOrAllocRuntimeTypeIndex();            \
  static int32_t RuntimeTypeIndex() { return TypeName::_type_index; }           \
  static int32_t _GetOrAllocRuntimeTypeIndex() {                                \
    return ::tvm::ffi::details::ObjectGetOrAllocTypeIndex(                      \
        TypeName::_type_key, -1, ParentType::_GetOrAllocRuntimeTypeIndex(),     \
        TypeName::_type_child_slots, TypeName::_type_child_slots_can_overflow); \
  }

/*
 * \brief Define object reference methods.
 * \param TypeName The object type name
 * \param ParentType The parent type of the objectref
 * \param ObjectName The type name of the object.
 */
#define TVM_FFI_DEFINE_NULLABLE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)           \
  TypeName() = default;                                                                        \
  explicit TypeName(::tvm::ffi::ObjectPtr<::tvm::ffi::Object> n) : ParentType(n) {}            \
  TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName)                                        \
  const ObjectName* operator->() const { return static_cast<const ObjectName*>(data_.get()); } \
  const ObjectName* get() const { return operator->(); }                                       \
  using ContainerType = ObjectName;

/*
 * \brief Define object reference methods that is not nullable.
 *
 * \param TypeName The object type name
 * \param ParentType The parent type of the objectref
 * \param ObjectName The type name of the object.
 */
#define TVM_FFI_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)        \
  explicit TypeName(::tvm::ffi::ObjectPtr<::tvm::ffi::Object> n) : ParentType(n) {}            \
  TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName)                                        \
  const ObjectName* operator->() const { return static_cast<const ObjectName*>(data_.get()); } \
  const ObjectName* get() const { return operator->(); }                                       \
  static constexpr bool _type_is_nullable = false;                                             \
  using ContainerType = ObjectName;

namespace details {
/*!
 * \brief Namespace to internally manipulate object class.
 * \note These functions are only supposed to be used by internal
 * implementations and not external users of the tvm::ffi
 */
struct ObjectInternal {
  // NOTE: get ffi header from an object
  static TVM_FFI_INLINE TVMFFIObject* GetHeader(Object* src) {
    return &(src->header_);
  }

  // Create ObjectPtr from unknowned ptr
  template <typename T>
  static TVM_FFI_INLINE ObjectPtr<T> ObjectPtrFromUnowned(Object* raw_ptr) {
    return tvm::ffi::ObjectPtr<T>(raw_ptr);
  }
  // Create objectptr by moving from an existing address of object and setting its
  // address to nullptr
  template <typename T>
  static TVM_FFI_INLINE ObjectPtr<T> MoveObjectPtrFromRValueRef(Object** ref) {
    ObjectPtr<T> ptr;
    ptr.data_ = *ref;
    *ref = nullptr;
    return ptr;
  }
};

// Code section that depends on dynamic components
#if TVM_FFI_ALLOW_DYN_TYPE
/*!
 * \brief Get the type index using type key.
 *
 *  When the function is first time called for a type,
 *  it will register the type to the type table in the runtime.
 *  If the static_tindex is TypeIndex::kDynamic, the function will
 *  allocate a runtime type index.
 *  Otherwise, we will populate the type table and return the static index.
 *
 * \param type_key the type key.
 * \param static_tindex Static type index if any, can be -1, which means this is a dynamic index
 * \param parent_tindex The index of the parent.
 * \param type_child_slots Number of slots reserved for its children.
 * \param type_child_slots_can_overflow Whether to allow child to overflow the slots.
 *
 * \return The allocated type index
 */
TVM_FFI_DLL int ObjectGetOrAllocTypeIndex(const char* type_key, int32_t static_tindex,
                                          int32_t parent_tindex, int32_t type_child_slots,
                                          bool type_child_slots_can_overflow);

/*!
 * \brief Check whether child type is derived from parent type.
 * \param child_type_index The candidate child type index.
 * \param parent_type_index The candidate parent type index.
 * \return the Check result.
 */
TVM_FFI_DLL bool ObjectDerivedFrom(int32_t child_type_index, int32_t parent_type_index);
#endif  // TVM_FFI_ALLOW_DYN_TYPE
}  // namespace details
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_OBJECT_H_
