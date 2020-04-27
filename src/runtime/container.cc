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
 * \file src/runtime/container.cc
 * \brief Implementations of common containers.
 */
#include <tvm/runtime/container.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

using namespace vm;

TVM_REGISTER_GLOBAL("runtime.GetADTTag")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectRef obj = args[0];
  const auto& adt = Downcast<ADT>(obj);
  *rv = static_cast<int64_t>(adt.tag());
});

TVM_REGISTER_GLOBAL("runtime.GetADTSize")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectRef obj = args[0];
  const auto& adt = Downcast<ADT>(obj);
  *rv = static_cast<int64_t>(adt.size());
});


TVM_REGISTER_GLOBAL("runtime.GetADTFields")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectRef obj = args[0];
  int idx = args[1];
  const auto& adt = Downcast<ADT>(obj);
  CHECK_LT(idx, adt.size());
  *rv = adt[idx];
});

TVM_REGISTER_GLOBAL("runtime.Tuple")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::vector<ObjectRef> fields;
  for (auto i = 0; i < args.size(); ++i) {
    fields.push_back(args[i]);
  }
  *rv = ADT::Tuple(fields);
});

TVM_REGISTER_GLOBAL("runtime.ADT")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  int itag = args[0];
  size_t tag = static_cast<size_t>(itag);
  std::vector<ObjectRef> fields;
  for (int i = 1; i < args.size(); i++) {
    fields.push_back(args[i]);
  }
  *rv = ADT(tag, fields);
});

TVM_REGISTER_GLOBAL("runtime.String")
.set_body_typed([](std::string str) {
  return String(std::move(str));
});

TVM_REGISTER_GLOBAL("runtime.GetFFIString")
.set_body_typed([](String str) {
  return std::string(str);
});

TVM_REGISTER_OBJECT_TYPE(ADTObj);
TVM_REGISTER_OBJECT_TYPE(StringObj);
TVM_REGISTER_OBJECT_TYPE(ClosureObj);

}  // namespace runtime
}  // namespace tvm
