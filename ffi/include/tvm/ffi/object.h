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

#include <tvm/ffi/c_ffi_abi.h>
#include <tvm/ffi/internal_utils.h>

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
 *       exceeds the _type_child_slots. A fallback mechanism to check global type table will be
 * used. Recommendation: set to false for optimal runtime speed if we know exact number of children.
 *
 * Two macros are used to declare helper functions in the object:
 * - Use TVM_FFI_DECLARE_BASE_OBJECT_INFO for object classes that can be sub-classed.
 * - Use TVM_FFI_DECLARE_FINAL_OBJECT_INFO for object classes that cannot be sub-classed.
 *
 * New objects can be created using make_object function.
 * Which will automatically populate the type_index and deleter of the object.
 */
class Object : private TVMFFIObject {
 public:
  Object() {
    TVMFFIObject::ref_counter = 0;
    TVMFFIObject::deleter = nullptr;
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
  static constexpr int32_t _type_index = TypeIndex::kTVMFFIDynObject;

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
  void IncRef() { details::AtomicIncrementRelaxed(&(this->ref_counter)); }

  /*! \brief decrease reference count and delete the object */
  void DecRef() {
    if (details::AtomicDecrementRelAcq(&(this->ref_counter)) == 1) {
      if (this->deleter != nullptr) {
        this->deleter(this);
      }
    }
  }

  /*!
   * \return The usage count of the cell.
   * \note We use stl style naming to be consistent with known API in shared_ptr.
   */
  int32_t use_count() const { return details::AtomicLoadRelaxed(&(this->ref_counter)); }

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
  /*!
   * \brief Move an ObjectPtr from an RValueRef argument.
   * \param ref The rvalue reference.
   * \return the moved result.
   */
  static ObjectPtr<T> MoveFromRValueRefArg(Object** ref) {
    ObjectPtr<T> ptr;
    ptr.data_ = *ref;
    *ref = nullptr;
    return ptr;
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

namespace details {
/*!
 * \brief Namespace to internally manipulate object class.
 * \note These functions are only supposed to be used by internal
 * implementations and not external users of the tvm::ffi
 */
struct ObjectInternal {
  // NOTE: these helper to perform static cast
  // that also allows conversion from/to FFI values
  template <typename T, typename U>
  static TVM_FFI_INLINE T StaticCast(U src) {
    return static_cast<T>(src);
  }

  template <typename T>
  static TVM_FFI_INLINE ObjectPtr<T> ObjectPtr(Object* raw_ptr) {
    return tvm::ffi::ObjectPtr<T>(raw_ptr);
  }
};
}  // namespace details
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_OBJECT_H_
