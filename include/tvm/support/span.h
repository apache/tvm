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
 * \file span.h
 * \brief Reimplementation of part of C++-20 style span.
 */
#ifndef TVM_SUPPORT_SPAN_H_
#define TVM_SUPPORT_SPAN_H_

#include <cstddef>
#include <iterator>
#include <vector>

namespace tvm {
namespace support {

template<class T, class W> //, std::enable_if_t<std::is_constructible<W, std::add_rvalue_reference<T>::value>::value> = true>
class Span {
 public:
  class iterator : public std::iterator<std::input_iterator_tag, W> {
   public:
    inline iterator(T* ptr, T* end) : ptr_{ptr}, end_{end} {
      CHECK_GE(end, ptr);
    }

    inline W operator*() {
      return W(*ptr_);
    }

    inline iterator& operator++() {
      if (ptr_ != end_) ptr_++;
      return *this;
    }

    inline bool operator==(iterator other) {
      return ptr_ == other.ptr_ && end_ == other.end_;
    }

    inline bool operator!=(iterator other) {
      return !(*this == other);
    }

   private:
    T* ptr_;
    T* end_;
  };

  inline Span(T* begin, int num_elements) : begin_{begin}, end_{begin + num_elements} {}
  inline Span(T* begin, T* end) : begin_{begin}, end_{end} {}

  inline iterator begin() {
    return iterator(begin_, end_);
  }

  inline iterator end() {
    return iterator(end_, end_);
  }

  inline W operator[](int i) {
    T* to_return = begin_ + i;
    ICHECK_LT(to_return, end_) << "Span access out of bounds: " << i;
    return W(*to_return);
  }

  inline operator std::vector<W>() {
    return std::vector<W>(begin(), end());
  }

 protected:
  T* begin_;
  T* end_;
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_SPAN_H_
