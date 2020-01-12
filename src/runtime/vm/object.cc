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
 * \file src/runtime/vm/object.cc
 * \brief VM related objects.
 */
#include <tvm/logging.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include "../runtime_base.h"

namespace tvm {
namespace runtime {
namespace vm {

Closure::Closure(size_t func_index, std::vector<ObjectRef> free_vars) {
  auto ptr = make_object<ClosureObj>();
  ptr->func_index = func_index;
  ptr->free_vars = std::move(free_vars);
  data_ = std::move(ptr);
}

TVM_REGISTER_OBJECT_TYPE(ClosureObj);
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
