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

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace metadata {

/*!
 * \brief Common base class for all Metadata.
 *
 * This class is used in the visitor classes as a internal check to ensure that verify that all
 * parts of the Metadata struct used in codegen are Metadata objects.
 */
class MetadataBaseNode : public ::tvm::runtime::Object {
 public:
  virtual const char* get_c_struct_name() const = 0;

  static constexpr const char* _type_key = "metadata.MetadataBaseNode";
  TVM_DECLARE_BASE_OBJECT_INFO(MetadataBaseNode, ::tvm::runtime::Object);
};

/*! \brief Reference class for the common MetadataBaseNode class. */
class MetadataBase : public ::tvm::runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MetadataBase, ::tvm::runtime::ObjectRef, MetadataBaseNode);
};

template <typename C, class Ref>
class ArrayAccessor;

/*! \brief An iterator implementation that lazily instantiates the C++ wrapping Metadata class. */
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

 private:
  size_t index_;
  const ArrayAccessor<C, Ref>* parent_;
};

/*! \brief A span-like class which permits access to Array fields with complex elements.
 * These array fields should be accessed from C++ using the Metadata wrapper classes. This class
 * lazily instantiates those wrappers as they are accessed.
 */
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

/*! \brief A specialization of ArrayAccessor for String.
 * This class is needed because the String constructor signature is different from the typical
 * Metadata subclass.
 */
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

/*! \brief Enumerates the primitive types which can be part of a Metadata instance.
 *
 * These are separate from TIR DataType because TIR does not model structs.
 */
enum MetadataKind : uint8_t {
  kUint64 = 0,
  kInt64 = 1,
  kBool = 2,
  kString = 3,
  kHandle = 4,
  kMetadata = 5,
};

/*! \brief Container for arrays in the metadata.
 *
 * Type information is needed when emitting arrays. This container augments the data field with
 * the necessary typing information.
 */
class MetadataArrayNode : public MetadataBaseNode {
 public:
  MetadataArrayNode(Array<ObjectRef> array, MetadataKind kind, const char* type_key)
      : array(::std::move(array)), kind{kind}, type_key{type_key} {}

  const char* get_c_struct_name() const final;

  std::string get_element_c_struct_name() const {
    CHECK(kind == MetadataKind::kMetadata)
        << "cannot get struct name for MetadataArray with kind=" << kind;
    constexpr int prefix_size = sizeof("metadata.") - 1;
    constexpr int suffix_size = sizeof("Node") - 1;
    std::string type_key_str(type_key);
    return std::string("TVM") +
           type_key_str.substr(prefix_size, type_key_str.size() - prefix_size - suffix_size);
  }

  Array<ObjectRef> array;

  /*! \brief Describes the storage class of the emitted struct member. */
  MetadataKind kind;

  /*! \brief When `kind` is Metadata, type_key of the MetadataBaseNode used with this array. */
  const char* type_key;

  static constexpr const char* _type_key = "metadata.MetadataArrayNode";
  TVM_DECLARE_BASE_OBJECT_INFO(MetadataArrayNode, MetadataBaseNode);
};

/*! \brief Reference class for MetadataArray. */
class MetadataArray : public MetadataBase {
 public:
  MetadataArray(Array<ObjectRef> array, MetadataKind kind, const char* struct_name);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MetadataArray, MetadataBase, MetadataArrayNode);
};

}  // namespace metadata
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_METADATA_BASE_H_
