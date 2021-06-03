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

#include "./base.h"

namespace tvm {
namespace runtime {

class ShapeTupleObj : public Object {
 public:
  using index_type = int64_t;
  std::vector<index_type> data;

  static constexpr const uint32_t _type_index = runtime::TypeIndex::kRuntimeShapeTuple;
  static constexpr const char* _type_key = "ShapeTuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShapeTupleObj, Object);
 private:
  template <typename Iterator>
  void Init(Iterator begin, Iterator end) {
    data = std::vector<index_type>(begin, end);
  }
  friend class ShapeTuple;
}; 

class ShapeTuple : public ObjectRef {
 public:
  using index_type = ShapeTupleObj::index_type;
  explicit ShapeTuple(std::vector<index_type> shape) : ShapeTuple(shape.begin(), shape.end()) {}
  template<typename Iterator>
  explicit ShapeTuple(Iterator begin, Iterator end) {
    auto ptr = make_object<ShapeTupleObj>();
    ptr->Init(begin, end);
    data_ = std::move(ptr);
  }
  explicit ShapeTuple(std::initializer_list<index_type> shape) : ShapeTuple(shape.begin(), shape.end()) {}

  index_type operator[](size_t idx) const { return operator->()->data[idx]; }
  size_t ndim() const { return operator->()->data.size(); }
  TVM_DEFINE_OBJECT_REF_METHODS(ShapeTuple, ObjectRef, ShapeTupleObj);
}; 

}  // namespace runtime

// expose the functions to the root namespace.
using runtime::ShapeTuple;
using runtime::ShapeTupleObj;
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_SHAPE_TUPLE_H_
