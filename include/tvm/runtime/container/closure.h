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
 * \file tvm/runtime/container/closure.h
 * \brief Runtime Closure container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_CLOSURE_H_
#define TVM_RUNTIME_CONTAINER_CLOSURE_H_

#include "./base.h"

namespace tvm {
namespace runtime {

/*!
 * \brief An object representing a closure. This object is used by both the
 * Relay VM and interpreter.
 */
class ClosureObj : public Object {
 public:
  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeClosure;
  static constexpr const char* _type_key = "runtime.Closure";
  TVM_DECLARE_BASE_OBJECT_INFO(ClosureObj, Object);
};

/*! \brief reference to closure. */
class Closure : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Closure, ObjectRef, ClosureObj);
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_CLOSURE_H_
