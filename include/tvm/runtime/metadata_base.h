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
 * \file tvm/runtime/metadata_base.h
 * \brief Defines types which can be used in Metadata.
 */
#ifndef TVM_RUNTIME_METADATA_BASE_H_
#define TVM_RUNTIME_METADATA_BASE_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/object.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace metadata {

class MetadataBaseNode : public ::tvm::runtime::Object {
 public:
  virtual std::string get_name() = 0;

  static constexpr const char* _type_key = "metadata.MetadataBaseNode";
  TVM_DECLARE_BASE_OBJECT_INFO(MetadataBaseNode, ::tvm::runtime::Object);
};

class MetadataBase : public ::tvm::runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MetadataBase, ::tvm::runtime::ObjectRef, MetadataBaseNode);
};

template <typename C, class Ref>
class ArrayAccessor;

template <typename C, class Ref>
class ArrayIterator {
 public:
  ArrayIterator(size_t index, const ArrayAccessor<C, Ref>* parent)
      : index_{index}, parent_{parent} {}

  inline Ref operator*() { return (*parent_)[index_]; }

  inline ArrayIterator<C, Ref>& operator++() {
    if (index_ < parent_->size()) {
      index_++;
    }

    return *this;
  }

  inline bool operator==(const ArrayIterator<C, Ref>& other) const {
    return parent_ == other.parent_ && index_ == other.index_;
  }

  inline bool operator!=(const ArrayIterator<C, Ref>& other) const { return !operator==(other); }

  // private:
  size_t index_;
  const ArrayAccessor<C, Ref>* parent_;
};

template <typename C, class Ref>
class ArrayAccessor {
 public:
  using value_type = Ref;
  using iterator = ArrayIterator<C, Ref>;
  using const_iterator = iterator;

  template <typename T = typename std::enable_if<std::is_base_of<ObjectRef, Ref>::value>::type>
  ArrayAccessor(const C* data, size_t num_data) : data_{data}, num_data_{num_data} {}

  inline size_t size() const { return num_data_; }

  inline Ref operator[](size_t index) const {
    if (index >= num_data_) {
      throw std::runtime_error("Index out of range");
    }

    return Ref(&data_[index]);
  }

  inline ArrayIterator<C, Ref> begin() const { return ArrayIterator<C, Ref>{0, this}; }

  inline ArrayIterator<C, Ref> end() const { return ArrayIterator<C, Ref>{num_data_, this}; }

 private:
  const C* data_;
  size_t num_data_;
};

template <>
class ArrayAccessor<const char*, ::tvm::runtime::String> {
 public:
  using value_type = ::tvm::runtime::String;
  using iterator = ArrayIterator<const char*, ::tvm::runtime::String>;
  using const_iterator = iterator;

  ArrayAccessor(const char** data, size_t num_data) : data_{data}, num_data_{num_data} {}

  inline size_t size() const { return num_data_; }

  inline ::tvm::runtime::String operator[](size_t index) const {
    if (index >= num_data_) {
      throw std::runtime_error("Index out of range");
    }
    return ::tvm::runtime::String(data_[index]);
  }

  inline ArrayIterator<const char*, ::tvm::runtime::String> begin() const {
    return ArrayIterator<const char*, ::tvm::runtime::String>{0, this};
  }

  inline ArrayIterator<const char*, ::tvm::runtime::String> end() const {
    return ArrayIterator<const char*, ::tvm::runtime::String>{num_data_, this};
  }

 private:
  const char** data_;
  size_t num_data_;
};

enum MetadataTypeIndex : uint8_t {
  kUint64 = 0,
  kInt64 = 1,
  kBool = 2,
  kString = 3,
  kHandle = 4,
  kMetadata = 5,
};

class MetadataArrayNode : public MetadataBaseNode {
 public:
  MetadataArrayNode(Array<ObjectRef> array, MetadataTypeIndex type_index, const char* struct_name)
      : array(::std::move(array)), type_index{type_index}, struct_name{struct_name} {}

  std::string get_name() override;

  Array<ObjectRef> array;
  MetadataTypeIndex type_index;
  const char* struct_name;
  static constexpr const char* _type_key = "metadata.MetadataArrayNode";
  TVM_DECLARE_BASE_OBJECT_INFO(MetadataArrayNode, MetadataBaseNode);
};

class MetadataArray : public MetadataBase {
 public:
  MetadataArray(Array<ObjectRef> array, MetadataTypeIndex type_index, const char* struct_name);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MetadataArray, MetadataBase, MetadataArrayNode);
};

}  // namespace metadata
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_METADATA_BASE_H_
