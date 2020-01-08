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

Tensor::Tensor(NDArray data) {
  auto ptr = make_object<TensorObj>();
  ptr->data = std::move(data);
  data_ = std::move(ptr);
}

Closure::Closure(size_t func_index, std::vector<ObjectRef> free_vars) {
  auto ptr = make_object<ClosureObj>();
  ptr->func_index = func_index;
  ptr->free_vars = std::move(free_vars);
  data_ = std::move(ptr);
}


TVM_REGISTER_GLOBAL("_vmobj.GetTensorData")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectRef obj = args[0];
  const auto* cell = obj.as<TensorObj>();
  CHECK(cell != nullptr);
  *rv = cell->data;
});

TVM_REGISTER_GLOBAL("_vmobj.GetADTTag")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectRef obj = args[0];
  const auto& adt = Downcast<ADT>(obj);
  *rv = static_cast<int64_t>(adt.tag());
});

TVM_REGISTER_GLOBAL("_vmobj.GetADTNumberOfFields")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectRef obj = args[0];
  const auto& adt = Downcast<ADT>(obj);
  *rv = static_cast<int64_t>(adt.size());
});


TVM_REGISTER_GLOBAL("_vmobj.GetADTFields")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectRef obj = args[0];
  int idx = args[1];
  const auto& adt = Downcast<ADT>(obj);
  CHECK_LT(idx, adt.size());
  *rv = adt[idx];
});

TVM_REGISTER_GLOBAL("_vmobj.Tensor")
.set_body([](TVMArgs args, TVMRetValue* rv) {
*rv = Tensor(args[0].operator NDArray());
});

TVM_REGISTER_GLOBAL("_vmobj.Tuple")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::vector<ObjectRef> fields;
  for (auto i = 0; i < args.size(); ++i) {
    fields.push_back(args[i]);
  }
  *rv = ADT::Tuple(fields);
});

TVM_REGISTER_GLOBAL("_vmobj.ADT")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  int itag = args[0];
  size_t tag = static_cast<size_t>(itag);
  std::vector<ObjectRef> fields;
  for (int i = 1; i < args.size(); i++) {
    fields.push_back(args[i]);
  }
  *rv = ADT(tag, fields);
});

TVM_REGISTER_OBJECT_TYPE(TensorObj);
TVM_REGISTER_OBJECT_TYPE(ADTObj);
TVM_REGISTER_OBJECT_TYPE(ClosureObj);
}  // namespace vm
}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

int TVMGetObjectTag(TVMObjectHandle handle, int* tag) {
  API_BEGIN();
  int res = static_cast<int>(static_cast<Object*>(handle)->type_index());
  *tag = res;
  API_END();
}
