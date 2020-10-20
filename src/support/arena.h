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
 * \brief Arena allocator that allocates
 *  memory chunks and frees them all during destruction time.
 */
#ifndef TVM_SUPPORT_ARENA_H_
#define TVM_SUPPORT_ARENA_H_

#include <cstddef>
#include <type_traits>
#include <utility>

#include "generic_arena.h"

namespace tvm {
namespace support {

/*!
 * \brief Simple page allocator that uses new and delete.
 */
class SimplePageAllocator {
 public:
  /*!
   * \brief Allocate a new page.
   * \param min_size Minimum size of the page.
   * \return The allocated page.
   * \note This function can return a bigger page to meet the min_size requirement.
   */
  ArenaPageHeader* allocate(size_t min_size) {
    size_t npages = ((min_size + kPageSize - 1) / kPageSize);
    ArenaPageHeader* header = reinterpret_cast<ArenaPageHeader*>(new Page[npages]);
    header->size = npages * kPageSize;
    header->offset = sizeof(ArenaPageHeader);
    return header;
  }
  /*!
   * \brief De-allocate an allocate page.
   * \param page The page to be de-allocated.
   */
  void deallocate(ArenaPageHeader* page) { delete[] reinterpret_cast<Page*>(page); }

  static const constexpr int kPageSize = 16 << 10;
  static const constexpr int kPageAlign = 1024;

 private:
  // page size 16 KB
  // The page data type;
  using Page = std::aligned_storage<kPageSize, kPageAlign>::type;
};

using Arena = GenericArena<SimplePageAllocator>;

/*!
 * \brief Link list node
 * \tparam T the content data type
 */
template <typename T>
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
template <typename T>
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

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_ARENA_H_
