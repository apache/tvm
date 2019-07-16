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
 *  Copyright (c) 2019 by Contributors
 * \file tvm/runtime/object.h
 * \brief A managed object in the TVM runtime.
 */
#ifndef TVM_RUNTIME_OBJECT_H_
#define TVM_RUNTIME_OBJECT_H_

#include <tvm/runtime/ndarray.h>
#include <memory>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

template <typename T>
class ObjectPtr;
class Object;

enum struct ObjectTag {
  /*! \brief The tag of a tensor. */
  kTensor = 0U,
  /*! \brief The tag of a closure. */
  kClosure = 1U,
  /*! \brief The tag of a structure. */
  kDatatype = 2U,
};

std::ostream& operator<<(std::ostream& os, const ObjectTag&);

struct ObjectCell {
 public:
  /*!
   * \brief The type of object deleter.
   * \param The self pointer to the ObjectCell.
   */
  typedef void (*FDeleter)(ObjectCell* self);

  /*! \brief The tag of the object.
   *
   * Describes which type of value
   * is represented by this object.
   */
  ObjectTag tag;

  /*!
   * \brief Increment the reference count.
   */
  void IncRef() { ref_counter_.fetch_add(1, std::memory_order_relaxed); }

  /*!
   * \brief Decrement the reference count.
   */
  void DecRef() {
    if (ref_counter_.fetch_sub(1, std::memory_order_release) == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      if (this->deleter_ != nullptr) {
        (*this->deleter_)(this);
      }
    }
  }

 protected:
  // default constructor and copy constructor
  ObjectCell() {}

  explicit ObjectCell(ObjectTag tag) : tag(tag) {}

  // override the copy and assign constructors to do nothing.
  // This is to make sure only contents, but not deleter and ref_counter
  // are copied when a child class copies itself.
  ObjectCell(const ObjectCell& other) {  // NOLINT(*)
  }

  ObjectCell(ObjectCell&& other) {  // NOLINT(*)
  }

  ObjectCell& operator=(const ObjectCell& other) {  // NOLINT(*)
    return *this;
  }

  ObjectCell& operator=(ObjectCell&& other) {  // NOLINT(*)
    return *this;
  }

 private:
  /*! \brief Internal reference counter */
  std::atomic<int> ref_counter_{0};
  /*!
   * \brief deleter of this object to enable customized allocation.
   * If the deleter is nullptr, no deletion will be performed.
   * The creator of the Node must always set the deleter field properly.
   */
  FDeleter deleter_ = nullptr;

  int use_count() const { return ref_counter_.load(std::memory_order_relaxed); }

  // friend declaration
  template <typename>
  friend class ObjectPtr;

  template <typename Y, typename... Args>
  friend ObjectPtr<Y> MakeObject(Args&&...);
};

/*!
 * \brief A custom smart pointer for Object.
 *  must be subclass of NodeBase
 * \tparam T the content data type.
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
   * \brief copy assignmemt
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
   * \brief move assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  ObjectPtr<T>& operator=(ObjectPtr<T>&& other) {  // NOLINT(*)
    // copy-and-swap idiom
    ObjectPtr(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }

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
  bool operator==(std::nullptr_t null) const { return data_ == nullptr; }

  /*! \return Whether the pointer is not nullptr */
  bool operator!=(std::nullptr_t null) const { return data_ != nullptr; }

  /* ObjectPtr's support custom allocators.
   *
   * The below allocator represents the simplest
   * possible impl. It can be easily swapped
   * for customized executor's, different allocation
   * strategies, and so on.
   *
   * See memory.h for more discussion on NodePtr's
   * allocator.
   */
  class StdAllocator {
   public:
    template <typename... Args>
    static T* New(Args&&... args) {
      return new T(std::forward<Args>(args)...);
    }

    static ObjectCell::FDeleter Deleter() { return Deleter_; }

   private:
    static void Deleter_(ObjectCell* ptr) { delete static_cast<T*>(ptr); }
  };

  template <typename U>
  ObjectPtr<U> As() const {
    auto ptr = reinterpret_cast<U*>(get());
    return ObjectPtr<U>(ptr);
  }

 private:
  /*! \brief internal pointer field */
  ObjectCell* data_{nullptr};
  /*!
   * \brief constructor from NodeBase
   * \param data The node base pointer
   */
  // TODO(jroesch): NodePtr design doesn't really work here due to the passing.
 public:
  explicit ObjectPtr(ObjectCell* data) : data_(data) {
    if (data != nullptr) {
      data_->IncRef();
    }
  }

 private:
  template <typename Y, typename... Args>
  friend ObjectPtr<Y> MakeObject(Args&&...);
  template <typename>
  friend class ObjectPtr;
  friend class NDArray;
  friend class TVMPODValue_;
  friend class TVMArgValue;
  friend class TVMRetValue;
  friend class RPCWrappedFunc;
};

struct TensorCell;
struct DatatypeCell;
struct ClosureCell;

/*!
 * \brief A managed object in the TVM runtime.
 *
 * For example a tuple, list, closure, and so on.
 *
 * Maintains a reference count for the object.
 */
class Object {
 public:
  ObjectPtr<ObjectCell> ptr_;
  explicit Object(ObjectPtr<ObjectCell> ptr) : ptr_(ptr) {}
  explicit Object(ObjectCell* ptr) : ptr_(ptr) {}
  Object() : ptr_() {}
  Object(const Object& obj) : ptr_(obj.ptr_) {}
  ObjectCell* operator->() { return this->ptr_.operator->(); }
  const ObjectCell* operator->() const { return this->ptr_.operator->(); }

  /*! \brief Construct a tensor object. */
  static Object Tensor(const NDArray& data);
  /*! \brief Construct a datatype object. */
  static Object Datatype(size_t tag, const std::vector<Object>& fields);
  /*! \brief Construct a tuple object. */
  static Object Tuple(const std::vector<Object>& fields);
  /*! \brief Construct a closure object. */
  static Object Closure(size_t func_index, const std::vector<Object>& free_vars);

  ObjectPtr<TensorCell> AsTensor() const;
  ObjectPtr<DatatypeCell> AsDatatype() const;
  ObjectPtr<ClosureCell> AsClosure() const;
};

/*! \brief An object containing an NDArray. */
struct TensorCell : public ObjectCell {
  /*! \brief The NDArray. */
  NDArray data;
  explicit TensorCell(const NDArray& data) : ObjectCell(ObjectTag::kTensor), data(data) {}
};

/*! \brief An object representing a structure or enumeration. */
struct DatatypeCell : public ObjectCell {
  /*! \brief The tag representing the constructor used. */
  size_t tag;
  /*! \brief The fields of the structure. */
  std::vector<Object> fields;

  DatatypeCell(size_t tag, const std::vector<Object>& fields)
      : ObjectCell(ObjectTag::kDatatype), tag(tag), fields(fields) {}
};

/*! \brief An object representing a closure. */
struct ClosureCell : public ObjectCell {
  /*! \brief The index into the VM function table. */
  size_t func_index;
  /*! \brief The free variables of the closure. */
  std::vector<Object> free_vars;

  ClosureCell(size_t func_index, const std::vector<Object>& free_vars)
      : ObjectCell(ObjectTag::kClosure), func_index(func_index), free_vars(free_vars) {}
};

/*! \brief Extract the NDArray from a tensor object. */
NDArray ToNDArray(const Object& obj);

/*!
 * \brief Allocate a node object.
 * \param args arguments to the constructor.
 * \tparam T the node type.
 * \return The NodePtr to the allocated object.
 */
template <typename T, typename... Args>
inline ObjectPtr<T> MakeObject(Args&&... args) {
  using Allocator = typename ObjectPtr<T>::StdAllocator;
  static_assert(std::is_base_of<ObjectCell, T>::value, "MakeObject can only be used to create ");
  T* node = Allocator::New(std::forward<Args>(args)...);
  node->deleter_ = Allocator::Deleter();
  return ObjectPtr<T>(node);
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OBJECT_H_
