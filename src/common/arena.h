/*!
 * Copyright 2018 by Contributors
 *
 * \file arena.h
 * \brief Arena allocator that allocates
 *  memory chunks and frees them all during destruction time.
 */
#ifndef TVM_COMMON_ARENA_H_
#define TVM_COMMON_ARENA_H_

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
   */
  template<typename T>
  T* Alloc() {
    return static_cast<T*>(Alloc(sizeof(T), alignof(T)));
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

}  // namespace common
}  // namespace tvm
#endif  // TVM_COMMON_ARENA_H_
