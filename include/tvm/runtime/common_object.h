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
 * \file tvm/runtime/common_object.h
 * \brief The objects that are commonly used by different runtime, i.e. Relay VM
 * and interpreter.
 */
#ifndef TVM_RUNTIME_COMMON_OBJECT_H_
#define TVM_RUNTIME_COMMON_OBJECT_H_

#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief An object representing a closure. This object is used by both the
 * Relay VM and interpreter.
 */
class ClosureObj : public Object {
 public:
  /*!
   * \brief The index into the function list. The function could be any
   * function object that is compatible to a certain runtime, i.e. VM or
   * interpreter.
   */
  size_t func_index;
  /*! \brief The free variables of the closure. */
  std::vector<ObjectRef> free_vars;

  static constexpr const uint32_t _type_index = TypeIndex::kClosure;
  static constexpr const char* _type_key = "Closure";
  TVM_DECLARE_FINAL_OBJECT_INFO(ClosureObj, Object);
};

/*! \brief reference to closure. */
class Closure : public ObjectRef {
 public:
  Closure(size_t func_index, std::vector<ObjectRef> free_vars) {
    auto ptr = make_object<ClosureObj>();
    ptr->func_index = func_index;
    ptr->free_vars = std::move(free_vars);
    data_ = std::move(ptr);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(Closure, ObjectRef, ClosureObj);
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_COMMON_OBJECT_H_
