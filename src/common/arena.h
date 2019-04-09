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
 * Copyright 2018 by Contributors
 *
 * \file arena.h
 * \brief Arena allocator that allocates
 *  memory chunks and frees them all during destruction time.
 */
#ifndef TVM_COMMON_ARENA_H_
#define TVM_COMMON_ARENA_H_

#include <utility>
#include <type_traits>

namespace tvm {
namespace common {

const constexpr int kArenaPageSize = 16 << 10;

/*!
 * \brief Arena allocator that allocates memory from continuous
 *  chunk and frees them all only during destruction.
 */
class Arena {
 public:
  Arena() {
    // eagerly allocate the first page.
    head_ = reinterpret_cast<PageHeader*>(new Page());
    head_->next = nullptr;
    head_->ptr = sizeof(PageHeader);
  }
  ~Arena() {
    // delete all the allocated pages.
    while (head_ != nullptr) {
      Page* page = reinterpret_cast<Page*>(head_);
      head_ = head_->next;
      delete page;
    }
  }
  /*!
   * \brief Allocate a space from Arena for type T
   * \param T the data type to be allocated
   * \note The space of T is not initialized.
   */
  template<typename T>
  T* allocate_() {
    return static_cast<T*>(Alloc(sizeof(T), alignof(T)));
  }
  /*!
   * \brief Create a new instance of type T.
   * \param args The constructor argument.
   * \tparam T the type to be created.
   * \tparam Args Arguments to the constructor.
   *
   * \return The allocated object.
   * \note The type T must be simple type, or only contain
   *  memory allocated from the same arena.
   *  Otherwise the destructor needs to be called explicitly.
   */
  template<typename T, typename... Args>
  T* make(Args&&... args) {
    T* ptr = allocate_<T>();
    new (ptr) T(std::forward<Args>(args)...);
    return ptr;
  }

 private:
  // page size 16 KB
  // The page data type;
  using Page = std::aligned_storage<kArenaPageSize, 1024>::type;
  /*! \brief Page header */
  struct PageHeader {
    /*! \brief points to the next page */
    PageHeader* next;
    /*! \brief memory allocator ptr inside page */
    size_t ptr;
  };
  /* \brief The page header */
  PageHeader* head_{nullptr};
  /*!
   * \brief Align ptr by upper bound.
   * \param ptr The pointer value.
   * \param align The alignment requirement.
   */
  size_t UpperAlign(size_t ptr, size_t align) {
    return ptr + (align - (ptr % align)) % align;
  }
  /*!
   * \brief Internal aligned alloc function.
   * \param size The size of the memory.
   * \param align The alignment requirement.
   */
  void* Alloc(size_t size, size_t align) {
    size_t ptr = UpperAlign(head_->ptr, align);
    if (ptr + size <= kArenaPageSize) {
      head_->ptr = ptr + size;
      return reinterpret_cast<char*>(head_) + ptr;
    } else {
      PageHeader* new_head = reinterpret_cast<PageHeader*>(new Page());
      new_head->next = head_;
      ptr = UpperAlign(sizeof(PageHeader), align);
      CHECK_LE(ptr + size, kArenaPageSize);
      new_head->ptr = ptr + size;
      head_ = new_head;
      return reinterpret_cast<char*>(head_) + ptr;
    }
  }
};

/*!
 * \brief Link list node
 * \tparam T the content data type
 */
template<typename T>
struct LinkNode {
  /*! \brief The content value */
  T value;
  /*! \brief pointer to the next location */
  LinkNode<T>* next{nullptr};
};
/*!
 * \brief LinkedList structure
 * \tparam T the content data type
 * \note This is a simple data structure that can be used together with the arena.
 * \sa LinkNode
 */
template<typename T>
struct LinkedList {
  /*! \brief Head pointer */
  LinkNode<T>* head{nullptr};
  /*! \brief Tail pointer */
  LinkNode<T>* tail{nullptr};
  /*!
   * \brief Push a new node to the end of the linked list.
   * \param node The node to be pushed.
   */
  void Push(LinkNode<T>* node) {
    node->next = nullptr;
    if (this->tail != nullptr) {
      this->tail->next = node;
      this->tail = node;
    } else {
      head = tail = node;
    }
  }
};

}  // namespace common
}  // namespace tvm
#endif  // TVM_COMMON_ARENA_H_
