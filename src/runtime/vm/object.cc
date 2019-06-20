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
 *  Copyright (c) 2019 by Contributors
 * \file object.cc
 * \brief A managed object in the TVM runtime.
 */

#include <tvm/logging.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <iostream>
#include "../runtime_base.h"

namespace tvm {
namespace runtime {

std::ostream& operator<<(std::ostream& os, const ObjectTag& tag) {
  switch (tag) {
    case ObjectTag::kClosure:
      os << "Closure";
      break;
    case ObjectTag::kDatatype:
      os << "Datatype";
      break;
    case ObjectTag::kTensor:
      os << "Tensor";
      break;
    default:
      LOG(FATAL) << "Invalid object tag: found " << static_cast<int>(tag);
  }
  return os;
}

Object Object::Tensor(const NDArray& data) {
  ObjectPtr<ObjectCell> ptr = MakeObject<TensorCell>(data);
  return Object(ptr);
}

Object Object::Datatype(size_t tag, const std::vector<Object>& fields) {
  ObjectPtr<ObjectCell> ptr = MakeObject<DatatypeCell>(tag, fields);
  return Object(ptr);
}

Object Object::Tuple(const std::vector<Object>& fields) { return Object::Datatype(0, fields); }

Object Object::Closure(size_t func_index, const std::vector<Object>& free_vars) {
  ObjectPtr<ObjectCell> ptr = MakeObject<ClosureCell>(func_index, free_vars);
  return Object(ptr);
}

ObjectPtr<TensorCell> Object::AsTensor() const {
  CHECK(ptr_.get());
  CHECK(ptr_.get()->tag == ObjectTag::kTensor);
  return ptr_.As<TensorCell>();
}

ObjectPtr<DatatypeCell> Object::AsDatatype() const {
  CHECK(ptr_.get());
  CHECK(ptr_.get()->tag == ObjectTag::kDatatype);
  return ptr_.As<DatatypeCell>();
}

ObjectPtr<ClosureCell> Object::AsClosure() const {
  CHECK(ptr_.get());
  CHECK(ptr_.get()->tag == ObjectTag::kClosure);
  return ptr_.As<ClosureCell>();
}

NDArray ToNDArray(const Object& obj) {
  auto tensor = obj.AsTensor();
  return tensor->data;
}

TVM_REGISTER_GLOBAL("_vmobj.GetTensorData")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  Object obj = args[0];
  auto cell = obj.AsTensor();
  *rv = cell->data;
});

TVM_REGISTER_GLOBAL("_vmobj.GetDatatypeTag")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  Object obj = args[0];
  auto cell = obj.AsDatatype();
  *rv = static_cast<int>(cell->tag);
});

TVM_REGISTER_GLOBAL("_vmobj.GetDatatypeNumberOfFields")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  Object obj = args[0];
  auto cell = obj.AsDatatype();
  *rv = static_cast<int>(cell->fields.size());
});


TVM_REGISTER_GLOBAL("_vmobj.GetDatatypeFields")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  Object obj = args[0];
  int idx = args[1];
  auto cell = obj.AsDatatype();
  CHECK_LT(idx, cell->fields.size());
  *rv = cell->fields[idx];
});

TVM_REGISTER_GLOBAL("_vmobj.Tensor")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = Object::Tensor(args[0]);
});

TVM_REGISTER_GLOBAL("_vmobj.Tuple")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::vector<Object> fields;
  for (auto i = 0; i < args.size(); ++i) {
    fields.push_back(args[i]);
  }
  *rv = Object::Tuple(fields);
});

TVM_REGISTER_GLOBAL("_vmobj.Datatype")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  int itag = args[0];
  size_t tag = static_cast<size_t>(itag);
  std::vector<Object> fields;
  for (int i = 1; i < args.size(); i++) {
    fields.push_back(args[i]);
  }
  *rv = Object::Datatype(tag, fields);
});

}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

int TVMGetObjectTag(TVMObjectHandle handle, int* tag) {
  API_BEGIN();
  *tag = static_cast<int>(static_cast<ObjectCell*>(handle)->tag);
  API_END();
}
