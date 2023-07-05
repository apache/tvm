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
 *
 * \file arena.h
 * \brief Arena allocator that allocates memory chunks and frees them all during destruction time.
 *
 * NOTE: This file is portable to bare-metal embedded devices. Don't use operator new (without
 * placement parameters) or malloc.
 */
#ifndef TVM_SUPPORT_GENERIC_ARENA_H_
#define TVM_SUPPORT_GENERIC_ARENA_H_

#ifndef TVM_ARENA_HAS_DESTRUCTOR
#define TVM_ARENA_HAS_DESTRUCTOR 1
#endif

#include <stddef.h>

#include <utility>

namespace tvm {
namespace support {

namespace {
template <typename T>              // For lvalues (T is T&),
T&& forward(T&& param) {           // take/return lvalue refs.
  return static_cast<T&&>(param);  // For rvalues (T is T),
}  // take/return rvalue refs.
}  // namespace

/*!
 * \brief An arena page header.
 */
struct ArenaPageHeader {
  /*! \brief points to the next page. */
  ArenaPageHeader* next;
  /*!
   * \brief Total size of the page.
   */
  size_t size;
  /*! \brief memory allocator offset inside page. */
  size_t offset;
};

/*!
 * \brief Arena allocator that allocates memory from continuous
 *  chunk and frees them all only during destruction.
 */
template <typename PageAllocator>
class GenericArena {
 public:
  explicit GenericArena(PageAllocator alloc = PageAllocator()) : alloc_(alloc) {
    // eagerly allocate the first page.
    head_ = tail_ = alloc_.allocate(1);
    head_->next = nullptr;
  }

#if TVM_ARENA_HAS_DESTRUCTOR
  ~GenericArena() { this->FreeAll(); }
#endif

  /*! \brief Free all pages. */
  void FreeAll() {
    FreePageList(&head_);
    FreePageList(&free_list_);
  }
  /*! \brief Recycle all the pages in the arena */
  void RecycleAll() {
    // put all the current list to the free list.
    tail_->next = free_list_;
    // allocate the first in the free list to head
    free_list_ = head_->next;
    head_->next = nullptr;
    // Reset the head.
    head_->offset = sizeof(ArenaPageHeader);
    tail_ = head_;
  }
  /*!
   * \brief Allocate a space from Arena for type T
   * \param T the data type to be allocated
   * \param count Numberof elements
   * \note The space of T is not initialized.
   */
  template <typename T>
  T* allocate_(int count = 1) {
    static_assert(PageAllocator::kPageAlign % alignof(T) == 0, "To large alignment");
    return static_cast<T*>(Alloc(sizeof(T) * count, alignof(T)));
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
  template <typename T, typename... Args>
  T* make(Args&&... args) {
    T* ptr = allocate_<T>();
    new (ptr) T(forward<Args>(args)...);
    return ptr;
  }

 private:
  /*! \brief internal page allocator. */
  PageAllocator alloc_;
  /* \brief The head of the allocated list. */
  ArenaPageHeader* head_{nullptr};
  /*! \brief The tail of the allocated list. */
  ArenaPageHeader* tail_{nullptr};
  /* \brief List of free pages. */
  ArenaPageHeader* free_list_{nullptr};
  /*!
   * \brief Align ptr by upper bound.
   * \param offset The offset value.
   * \param align The alignment requirement.
   */
  size_t UpperAlign(size_t offset, size_t align) {
    return offset + (align - (offset % align)) % align;
  }
  /*!
   * \brief Internal aligned alloc function.
   * \param size The size of the memory.
   * \param align The alignment requirement.
   */
  void* Alloc(size_t size, size_t align) {
    size_t offset = UpperAlign(head_->offset, align);
    if (offset + size <= head_->size) {
      head_->offset = offset + size;
      return reinterpret_cast<char*>(head_) + offset;
    } else {
      ArenaPageHeader* new_head;
      offset = UpperAlign(sizeof(ArenaPageHeader), align);
      if (free_list_ != nullptr && offset + size <= free_list_->size) {
        new_head = free_list_;
        free_list_ = free_list_->next;
      } else {
        new_head = alloc_.allocate(offset + size);
      }
      new_head->next = head_;
      new_head->offset = offset + size;
      head_ = new_head;
      return reinterpret_cast<char*>(head_) + offset;
    }
  }
  /*!
   * \brief Free all the pages in the list.
   * \param ptr The head ptr.
   */
  void FreePageList(ArenaPageHeader** ptr) {
    // delete all the allocated pages.
    while (ptr[0] != nullptr) {
      ArenaPageHeader* temp = ptr[0];
      ptr[0] = ptr[0]->next;
      alloc_.deallocate(temp);
    }
  }
};

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_GENERIC_ARENA_H_
