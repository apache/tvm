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

#ifndef TVM_RUNTIME_MICRO_STANDALONE_MINIMAL_VECTOR_H_
#define TVM_RUNTIME_MICRO_STANDALONE_MINIMAL_VECTOR_H_

#include <algorithm>
#include <cassert>
#include <memory>

namespace tvm {
namespace micro {

// A minimal wrapper, derived from https://github.com/Robbepop/dynarray/, that
// supports a minimal subset of the std::vector API with a minimized code size.
template <typename T>
struct DynArray {
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = value_type const&;
  using pointer = value_type*;
  using const_pointer = value_type const*;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  explicit DynArray(size_type size = 0) { resize(size); }

  DynArray(const DynArray& other) {
    resize(other.size());
    std::copy(other.begin(), other.end(), begin());
  }

  DynArray& operator=(const DynArray& other) {
    resize(other.size());
    std::copy(other.begin(), other.end(), begin());
    return *this;
  }

  void resize(size_type size) {
    if (size > 0) {
      data_.reset(new T[size]);
    } else {
      data_.reset();
    }
    size_ = size;
  }

  size_type size() const { return size_; }

  reference operator[](size_type pos) { return data_[pos]; }

  const_reference operator[](size_type pos) const { return data_[pos]; }

  pointer data() { return data_.get(); }

  const_pointer data() const { return data_.get(); }

  iterator begin() { return data_.get(); }

  const_iterator begin() const { return data_.get(); }

  const_iterator cbegin() const { return data_.get(); }

  iterator end() { return data_.get() + size_; }

  const_iterator end() const { return data_.get() + size_; }

  const_iterator cend() const { return data_.get() + size_; }

  reference front() { return data_[0]; }

  const_reference front() const { return data_[0]; }

  reference back() { return data_[size_ - 1]; }

  const_reference back() const { return data_[size_ - 1]; }

 private:
  std::unique_ptr<T[]> data_;
  size_type size_;
};

}  // namespace micro
}  // namespace tvm

#endif  // TVM_RUNTIME_MICRO_STANDALONE_MINIMAL_VECTOR_H_
