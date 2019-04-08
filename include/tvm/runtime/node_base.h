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
 * \file tvm/runtime/node_base.h
 * \brief Base data structure for Node.
 *
 * \note Node is not a runtime feature.
 *  This file only exposes the signature of NodePtr for PackedFunc.
 */
#ifndef TVM_RUNTIME_NODE_BASE_H_
#define TVM_RUNTIME_NODE_BASE_H_

#include <utility>
#include <atomic>

namespace tvm {

// forward declarations
template<typename T>
class NodePtr;
class Node;
class NodeRef;

/*!
 * \brief Base class of Node for runtime destructor purposes.
 *
 * Node is a reference counted object which is used to construct AST.
 * Each node is backed by a custom deleter, which deletes the object.
 * Do not call create raw Node pointer, always use tvm::make_node.
 *
 * \note In most cases, please inheritate tvm::Node.
 * \sa Node, NodePtr, make_node
 */
class NodeBase {
 public:
  /*!
   * \brief type of NodeBase deleter
   * \param self pointer to the NodeBase.
   */
  typedef void (*FDeleter)(NodeBase* self);

 protected:
  // default constructor and copy constructor
  NodeBase() {}
  // override the copy and assign constructors to do nothing.
  // This is to make sure only contents, but not deleter and ref_counter
  // are copied when a child class copies itself.
  NodeBase(const NodeBase& other) {  // NOLINT(*)
  }
  NodeBase(NodeBase&& other) {  // NOLINT(*)
  }
  NodeBase& operator=(const NodeBase& other) {  //NOLINT(*)
    return *this;
  }
  NodeBase& operator=(NodeBase&& other) {  //NOLINT(*)
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
  // reference counting functions
  void IncRef() {
    ref_counter_.fetch_add(1, std::memory_order_relaxed);
  }
  void DecRef() {
    if (ref_counter_.fetch_sub(1, std::memory_order_release) == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      if (this->deleter_ != nullptr) {
        (*this->deleter_)(this);
      }
    }
  }
  int use_count() const {
    return ref_counter_.load(std::memory_order_relaxed);
  }
  // friend declaration
  template<typename>
  friend class NodePtr;
  template<typename Y, typename... Args>
  friend NodePtr<Y> make_node(Args&&...);
};

/*!
 * \brief Smart pointer for Node containers,
 *  must be subclass of NodeBase
 * \tparam T the content data type.
 */
template<typename T>
class NodePtr {
 public:
  /*! \brief default constructor */
  NodePtr() {}
  /*! \brief default constructor */
  NodePtr(std::nullptr_t) {}  // NOLINT(*)
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  NodePtr(const NodePtr<T>& other)  // NOLINT(*)
      : NodePtr(other.data_) {
  }
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  template<typename Y>
  NodePtr(const NodePtr<Y>& other)  // NOLINT(*)
      : NodePtr(other.data_) {
    static_assert(std::is_base_of<T, Y>::value,
                  "can only assign of child class NodePtr to parent");
  }
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  NodePtr(NodePtr<T>&& other) // NOLINT(*)
      : data_(other.data_) {
    other.data_ = nullptr;
  }
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  template<typename Y>
  NodePtr(NodePtr<Y>&& other)  // NOLINT(*)
      : data_(other.data_) {
    static_assert(std::is_base_of<T, Y>::value,
                  "can only assign of child class NodePtr to parent");
    other.data_ = nullptr;
  }
  /*! \brief destructor */
  ~NodePtr() {
    this->reset();
  }
  /*!
   * \brief Swap this array with another NDArray
   * \param other The other NDArray
   */
  void swap(NodePtr<T>& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }
  /*!
   * \return Get the content of the pointer
   */
  T* get() const {
    return static_cast<T*>(data_);
  }
  /*!
   * \return The pointer
   */
  T* operator->() const {
    return get();
  }
  /*!
   * \return The reference
   */
  T& operator*() const { // NOLINT(*)
    return *get();
  }
  /*!
   * \brief copy assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  NodePtr<T>& operator=(const NodePtr<T>& other) {  // NOLINT(*)
    // takes in plane operator to enable copy elison.
    // copy-and-swap idiom
    NodePtr(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*!
   * \brief move assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  NodePtr<T>& operator=(NodePtr<T>&& other) {  // NOLINT(*)
    // copy-and-swap idiom
    NodePtr(std::move(other)).swap(*this); // NOLINT(*)
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
  int use_count() const {
    return data_ != nullptr ? data_->use_count() : 0;
  }
  /*! \return whether the reference is unique */
  bool unique() const {
    return data_ != nullptr && data_->use_count() == 1;
  }
  /*! \return Whether two NodePtr do not equals each other */
  bool operator==(const NodePtr<T>& other) const {
    return data_ == other.data_;
  }
  /*! \return Whether two NodePtr equals each other */
  bool operator!=(const NodePtr<T>& other) const {
    return data_ != other.data_;
  }
  /*! \return Whether the pointer is nullptr */
  bool operator==(std::nullptr_t null) const {
    return data_ == nullptr;
  }
  /*! \return Whether the pointer is not nullptr */
  bool operator!=(std::nullptr_t null) const {
    return data_ != nullptr;
  }

 private:
  /*! \brief internal pointer field */
  NodeBase* data_{nullptr};
  /*!
   * \brief constructor from NodeBase
   * \param data The node base pointer
   */
  explicit NodePtr(NodeBase* data)
      : data_(data) {
    if (data != nullptr) {
      data_->IncRef();
    }
  }
  // friend declaration
  friend class Node;
  template<typename>
  friend class NodePtr;
  template<typename Y, typename... Args>
  friend NodePtr<Y> make_node(Args&&...);
};
}  // namespace tvm

#endif  // TVM_RUNTIME_NODE_BASE_H_
