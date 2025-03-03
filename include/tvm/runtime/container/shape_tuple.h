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
 * \file tvm/runtime/container/shape_tuple.h
 * \brief Runtime ShapeTuple container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_SHAPE_TUPLE_H_
#define TVM_RUNTIME_CONTAINER_SHAPE_TUPLE_H_

#include <ostream>
#include <utility>
#include <vector>

#include "./base.h"

namespace tvm {
namespace runtime {

/*! \brief An object representing a shape tuple. */
class ShapeTupleObj : public Object {
 public:
  /*! \brief The type of shape index element. */
  using index_type = int64_t;
  /*! \brief The pointer to shape tuple data. */
  index_type* data;
  /*! \brief The size of the shape tuple object. */
  uint64_t size;

  /*! \brief Get "numel", meaning the number of elements of an array if the array has this shape */
  index_type Product() const;

  static constexpr const uint32_t _type_index = runtime::TypeIndex::kRuntimeShapeTuple;
  static constexpr const char* _type_key = "runtime.ShapeTuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShapeTupleObj, Object);

 private:
  /*! \brief ShapeTuple object which is moved from std::vector container. */
  class FromStd;

  friend class ShapeTuple;
};

/*! \brief An object representing shape tuple moved from std::vector. */
class ShapeTupleObj::FromStd : public ShapeTupleObj {
 public:
  /*! \brief The type of shape index element. */
  using index_type = ShapeTupleObj::index_type;
  /*!
   * \brief Construct a new FromStd object
   *
   * \param other The moved/copied std::vector object
   *
   * \note If user passes const reference, it will trigger copy. If it's rvalue,
   * it will be moved into other.
   */
  explicit FromStd(std::vector<index_type> other) : data_container{other} {}

 private:
  /*! \brief Container that holds the memory. */
  std::vector<index_type> data_container;

  friend class ShapeTuple;
};

/*!
 * \brief Reference to shape tuple objects.
 */
class ShapeTuple : public ObjectRef {
 public:
  /*! \brief The type of shape index element. */
  using index_type = ShapeTupleObj::index_type;

  /*!
   * \brief Construct an empty shape tuple.
   */
  ShapeTuple() : ShapeTuple(std::vector<index_type>()) {}

  /*!
   * \brief Constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  ShapeTuple(IterType begin, IterType end) : ShapeTuple(std::vector<index_type>(begin, end)) {}

  /*!
   * \brief constructor from initializer list
   * \param shape The initializer list
   */
  ShapeTuple(std::initializer_list<index_type> shape) : ShapeTuple(shape.begin(), shape.end()) {}

  /*!
   * \brief Construct a new ShapeTuple object
   *
   * \param shape The moved/copied std::vector object
   *
   * \note If user passes const reference, it will trigger copy. If it's rvalue,
   * it will be moved into other.
   */
  ShapeTuple(std::vector<index_type> shape);  // NOLINT(*)

  /*!
   * \brief Return the data pointer
   *
   * \return const index_type* data pointer
   */
  const index_type* data() const { return get()->data; }

  /*!
   * \brief Return the size of the shape tuple
   *
   * \return size_t shape tuple size
   */
  size_t size() const { return get()->size; }

  /*!
   * \brief Immutably read i-th element from the shape tuple.
   * \param idx The index
   * \return the i-th element.
   */
  index_type operator[](size_t idx) const {
    ICHECK(idx < this->size()) << "IndexError: indexing " << idx << " on an array of size "
                               << this->size();
    return this->data()[idx];
  }

  /*!
   * \brief Immutably read i-th element from the shape tuple.
   * \param idx The index
   * \return the i-th element.
   */
  index_type at(size_t idx) const { return this->operator[](idx); }

  /*! \return Whether shape tuple is empty */
  bool empty() const { return size() == 0; }

  /*! \return The first element of the shape tuple */
  index_type front() const { return this->at(0); }

  /*! \return The last element of the shape tuple */
  index_type back() const { return this->at(this->size() - 1); }

  /*! \return begin iterator */
  const index_type* begin() const { return get()->data; }

  /*! \return end iterator */
  const index_type* end() const { return (get()->data + size()); }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ShapeTuple, ObjectRef, ShapeTupleObj);
};

inline ShapeTuple::ShapeTuple(std::vector<index_type> shape) {
  auto ptr = make_object<ShapeTupleObj::FromStd>(std::move(shape));
  ptr->size = ptr->data_container.size();
  ptr->data = ptr->data_container.data();
  data_ = std::move(ptr);
}

inline ShapeTupleObj::index_type ShapeTupleObj::Product() const {
  index_type numel = 1;
  for (int i = 0, n = this->size; i < n; ++i) {
    numel *= this->data[i];
  }
  return numel;
}

inline std::ostream& operator<<(std::ostream& os, const ShapeTuple& shape) {
  os << '[';
  for (size_t i = 0; i < shape->size; ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << shape->data[i];
  }
  os << ']';
  return os;
}

using IntTuple = ShapeTuple;
using IntTupleObj = ShapeTupleObj;

}  // namespace runtime

// expose the functions to the root namespace.
using runtime::IntTuple;
using runtime::IntTupleObj;
using runtime::ShapeTuple;
using runtime::ShapeTupleObj;
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_SHAPE_TUPLE_H_
