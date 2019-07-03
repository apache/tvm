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
#include <iostream>

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

}  // namespace runtime
}  // namespace tvm
