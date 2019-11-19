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
 * \file tvm/runtime/container.h
 * \brief Common POD(plain old data) container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_H_
#define TVM_RUNTIME_CONTAINER_H_
#include <dmlc/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>

#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Base template for classes with array like memory layout.
 *
 *        It provides general methods to access the memory. The memory
 *        layout is ArrayType + [ElemType]. The alignment of ArrayType
 *        and ElemType is handled by the memory allocator.
 *
 * \tparam ArrayType
 * \tparam ElemType
 */
template <typename ArrayType, typename ElemType>
class InplaceArrayBase {
 public:
  /*!
   * \brief Initialize the elements in the array.
   */
  void Init() {
    CHECK_EQ(sizeof(ArrayType) % alignof(ElemType), 0);
    for (size_t i = 0; i < Self()->size(); ++i) {
      void* field_ptr = AddressOf(i);
      new (field_ptr) ElemType();
    }
  }

  /*!
   * \brief Initialize the elements in the array.
   *
   * \tparam Iterator Iterator type of the array.
   * \param begin The begin iterator.
   * \param end The end iterator.
   */
  template <typename Iterator>
  void Init(Iterator begin, Iterator end) {
    CHECK_EQ(sizeof(ArrayType) % alignof(ElemType), 0);
    ArrayType* self = Self();
    size_t num_elems = std::distance(begin, end);
    if (num_elems != self->size()) {
      LOG(FATAL)
          << "Number of initializer values does not match number of elements\n";
    }
    auto it = begin;
    for (size_t i = 0; i < num_elems; ++i) {
      void* field_ptr = AddressOf(i);
      new (field_ptr) ElemType(*it);
      ++it;
    }
  }

  /*!
   * \brief Initialize the elements in the array.
   *
   * \param init The initializer list of elements.
   */
  void Init(std::initializer_list<ElemType> init) {
    CHECK_EQ(sizeof(ArrayType) % alignof(ElemType), 0);
    Init(init.begin(), init.end());
  }

  /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Const reference to ElemType at the index.
   */
  const ElemType& operator[](size_t idx) const {
    size_t size = Self()->size();
    CHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
    return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
  }

  /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Reference to ElemType at the index.
   */
  ElemType& operator[](size_t idx) {
    return this->operator[](idx);
  }

  /*!
   * \brief Destroy the Inplace Array Base object
   */
  ~InplaceArrayBase() {
    if (!IsPOD()) {
      size_t size = Self()->size();
      for (size_t i = 0; i < size; ++i) {
        ElemType* fp = reinterpret_cast<ElemType*>(AddressOf(i));
        fp->ElemType::~ElemType();
      }
    }
  }

 private:
  /*!
   * \brief If the ElemType is Plain Old Data.
   */
  inline bool IsPOD() const {
    return std::is_standard_layout<ElemType>::value &&
           std::is_trivial<ElemType>::value;
  }

  /*!
   * \brief Return the self object for the array.
   *
   * \return Pointer to ArrayType.
   */
  inline ArrayType* Self() const {
    return static_cast<ArrayType*>(const_cast<InplaceArrayBase*>(this));
  }

  /*!
   * \brief Return the raw pointer to the element at idx.
   *
   * \param idx The index of the element.
   * \return Raw pointer to the element.
   */
  void* AddressOf(size_t idx) const {
    size_t kDataStart = sizeof(ArrayType);
    ArrayType* self = Self();
    char* data_start = reinterpret_cast<char*>(self) + kDataStart;
    return data_start + idx * sizeof(ElemType);
  }
};

/*! \brief An object representing a structure or enumeration. */
class ADTObj : public Object, public InplaceArrayBase<ADTObj, ObjectRef> {
 public:
  /*! \brief The tag representing the constructor used. */
  uint32_t tag_;
  /*! \brief Number of fields in the ADT object. */
  uint32_t size_;
  // The fields of the structure follows directly in memory.

  /*!
   * \brief The number of elements in the array.
   */
  inline size_t size() const { return size_; }

  static constexpr const uint32_t _type_index = TypeIndex::kVMADT;
  static constexpr const char* _type_key = "vm.ADT";
  TVM_DECLARE_FINAL_OBJECT_INFO(ADTObj, Object);
};

/*! \brief reference to algebraic data type objects. */
class ADT : public ObjectRef {
 public:
  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param fields The fields of the ADT object.
   * \return The constructed ADT object reference.
   */
  ADT(uint32_t tag, std::vector<ObjectRef> fields)
      : ADT(tag, fields.begin(), fields.end()){};

  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param begin The begin iterator to the start of the fields array.
   * \param end The end iterator to the end of the fields array.
   * \return The constructed ADT object reference.
   */
  template <typename Iterator>
  ADT(uint32_t tag, Iterator begin, Iterator end) {
    size_t num_elems = std::distance(begin, end);
    auto ptr = make_inplace_array_object<ADTObj, ObjectRef>(num_elems);
    ptr->tag_ = tag;
    ptr->size_ = num_elems;
    ptr->Init(begin, end);
    data_ = std::move(ptr);
  }

  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param init The initializer list of fields.
   * \return The constructed ADT object reference.
   */
  ADT(uint32_t tag, std::initializer_list<ObjectRef> init)
      : ADT(tag, init.begin(), init.end()){};

  /*!
   * \brief construct a tuple object.
   * \param fields The fields of the tuple.
   * \return The constructed tuple type.
   */
  static ADT Tuple(std::vector<ObjectRef> fields);

  TVM_DEFINE_OBJECT_REF_METHODS(ADT, ObjectRef, ADTObj);
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_H_
