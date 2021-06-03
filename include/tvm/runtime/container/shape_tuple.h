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

#include <utility>
#include <vector>

#include "./base.h"

namespace tvm {
namespace runtime {

class ShapeTupleObj : public Object {
 public:
  using index_type = int64_t;
  index_type* data;
  uint64_t ndim;

  static constexpr const uint32_t _type_index = runtime::TypeIndex::kRuntimeShapeTuple;
  static constexpr const char* _type_key = "runtime.ShapeTuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShapeTupleObj, Object);

 private:
  class FromStd;
  friend class ShapeTuple;
};

class ShapeTupleObj::FromStd : public ShapeTupleObj {
 public:
  using index_type = ShapeTupleObj::index_type;
  explicit FromStd(std::vector<index_type> other) : data_container{other} {}

 private:
  /*! \brief Container that holds the memory. */
  std::vector<index_type> data_container;

  friend class ShapeTuple;
};

class ShapeTuple : public ObjectRef {
 public:
  using index_type = ShapeTupleObj::index_type;
  ShapeTuple() : ShapeTuple(std::vector<index_type>()) {}
  template <typename Iterator>
  ShapeTuple(Iterator begin, Iterator end) : ShapeTuple(std::vector<index_type>(begin, end)) {}
  ShapeTuple(std::initializer_list<index_type> shape) : ShapeTuple(shape.begin(), shape.end()) {}

  ShapeTuple(std::vector<index_type> shape);  // NOLINT(*)

  index_type operator[](size_t idx) const { return get()->data[idx]; }
  index_type at(size_t idx) const { return get()->data[idx]; }
  index_type* data() { return get()->data; }
  size_t ndim() const { return get()->ndim; }
  size_t size() const { return get()->ndim; }

  index_type* begin() { return get()->data; }
  index_type* end() { return (get()->data + ndim()); }
  const index_type* begin() const { return get()->data; }
  const index_type* end() const { return (get()->data + ndim()); }
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ShapeTuple, ObjectRef, ShapeTupleObj);
};

inline ShapeTuple::ShapeTuple(std::vector<index_type> shape) {
  auto ptr = make_object<ShapeTupleObj::FromStd>(std::move(shape));
  ptr->ndim = ptr->data_container.size();
  ptr->data = ptr->data_container.data();
  data_ = std::move(ptr);
}

}  // namespace runtime

// expose the functions to the root namespace.
using runtime::ShapeTuple;
using runtime::ShapeTupleObj;
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_SHAPE_TUPLE_H_
