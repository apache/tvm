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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

// Array
TVM_REGISTER_OBJECT_TYPE(ArrayObj);

TVM_REGISTER_GLOBAL("runtime.Array").set_body_packed([](ffi::PackedArgs args, Any* ret) {
  Array<Any> result;
  for (int i = 0; i < args.size(); ++i) {
    result.push_back(args[i]);
  }
  *ret = result;
});

TVM_REGISTER_GLOBAL("runtime.ArrayGetItem")
    .set_body_typed([](const ffi::ArrayObj* n, int64_t i) -> Any { return n->at(i); });

TVM_REGISTER_GLOBAL("runtime.ArraySize").set_body_typed([](const ffi::ArrayObj* n) -> int64_t {
  return static_cast<int64_t>(n->size());
});

// String
TVM_REGISTER_GLOBAL("runtime.String").set_body_typed([](std::string str) {
  return String(std::move(str));
});

TVM_REGISTER_GLOBAL("runtime.GetFFIString").set_body_typed([](String str) {
  return std::string(str);
});

// Map
TVM_REGISTER_GLOBAL("runtime.Map").set_body_packed([](ffi::PackedArgs args, Any* ret) {
  ICHECK_EQ(args.size() % 2, 0);
  Map<Any, Any> data;
  for (int i = 0; i < args.size(); i += 2) {
    data.Set(args[i], args[i + 1]);
  }
  *ret = data;
});

TVM_REGISTER_GLOBAL("runtime.MapSize").set_body_typed([](const ffi::MapObj* n) -> int64_t {
  return static_cast<int64_t>(n->size());
});

TVM_REGISTER_GLOBAL("runtime.MapGetItem")
    .set_body_typed([](const ffi::MapObj* n, const Any& k) -> Any { return n->at(k); });

TVM_REGISTER_GLOBAL("runtime.MapCount")
    .set_body_typed([](const ffi::MapObj* n, const Any& k) -> int64_t { return n->count(k); });

TVM_REGISTER_GLOBAL("runtime.MapItems").set_body_typed([](const ffi::MapObj* n) -> Array<Any> {
  Array<Any> rkvs;
  for (const auto& kv : *n) {
    rkvs.push_back(kv.first);
    rkvs.push_back(kv.second);
  }
  return rkvs;
});

TVM_REGISTER_GLOBAL("runtime.GetShapeSize").set_body_typed([](ffi::Shape shape) {
  return static_cast<int64_t>(shape.size());
});

TVM_REGISTER_GLOBAL("runtime.GetShapeElem").set_body_typed([](ffi::Shape shape, int idx) {
  ICHECK_LT(idx, shape.size());
  return shape[idx];
});

}  // namespace runtime
}  // namespace tvm
