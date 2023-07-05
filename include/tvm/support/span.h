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
 * \file tvm/support/span.h
 * \brief Reimplementation of part of C++-20 style span.
 */
#ifndef TVM_SUPPORT_SPAN_H_
#define TVM_SUPPORT_SPAN_H_

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

namespace tvm {
namespace support {

/*!
 * \brief A partial implementation of the C++20 std::span.
 *
 * At the time of writing, TVM must compile against C++17.
 */
template <class T, class W>
class Span {
 public:
  using value_type = W;
  using const_W = typename std::add_const<W>::type;

  template <class W1>
  class iterator_base {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = W;
    using difference_type = std::ptrdiff_t;
    using pointer = const W*;
    using reference = const W&;

    inline iterator_base(T* ptr, T* end) : ptr_{ptr}, end_{end} { CHECK_GE(end, ptr); }

    inline W1 operator*() { return W1(*ptr_); }

    inline iterator_base<W1>& operator++() {
      if (ptr_ != end_) ptr_++;
      return *this;
    }

    inline bool operator==(iterator_base<W1> other) {
      return ptr_ == other.ptr_ && end_ == other.end_;
    }

    inline bool operator!=(iterator_base<W1> other) { return !(*this == other); }

    template <class X = W1, typename = std::enable_if_t<!std::is_const<X>::value>>
    inline operator iterator_base<const_W>() const {
      return iterator_base<const_W>(ptr_, end_);
    }

   private:
    T* ptr_;
    T* end_;
  };

  using iterator = iterator_base<W>;
  using const_iterator = iterator_base<const_W>;

  inline Span(T* begin, int num_elements) : begin_{begin}, end_{begin + num_elements} {}
  inline Span(T* begin, T* end) : begin_{begin}, end_{end} {}

  inline iterator begin() const { return iterator(begin_, end_); }

  inline iterator end() const { return iterator(end_, end_); }

  size_t size() const { return end_ - begin_; }

  inline W operator[](int i) {
    T* to_return = begin_ + i;
    ICHECK_LT(to_return, end_) << "Span access out of bounds: " << i;
    return W(*to_return);
  }

  inline operator std::vector<W>() { return std::vector<W>(begin(), end()); }

 protected:
  T* begin_;
  T* end_;
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_SPAN_H_
