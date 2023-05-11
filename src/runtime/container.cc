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
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/closure.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

// Array
TVM_REGISTER_OBJECT_TYPE(ArrayNode);

TVM_REGISTER_GLOBAL("runtime.Array").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::vector<ObjectRef> data;
  for (int i = 0; i < args.size(); ++i) {
    if (args[i].type_code() != kTVMNullptr) {
      data.push_back(args[i].operator ObjectRef());
    } else {
      data.push_back(ObjectRef(nullptr));
    }
  }
  *ret = Array<ObjectRef>(data);
});

TVM_REGISTER_GLOBAL("runtime.ArrayGetItem").set_body([](TVMArgs args, TVMRetValue* ret) {
  int64_t i = args[1];
  ICHECK_EQ(args[0].type_code(), kTVMObjectHandle);
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  ICHECK(ptr->IsInstance<ArrayNode>());
  auto* n = static_cast<const ArrayNode*>(ptr);
  ICHECK_LT(static_cast<size_t>(i), n->size()) << "out of bound of array";
  *ret = n->at(i);
});

TVM_REGISTER_GLOBAL("runtime.ArraySize").set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_EQ(args[0].type_code(), kTVMObjectHandle);
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  ICHECK(ptr->IsInstance<ArrayNode>());
  *ret = static_cast<int64_t>(static_cast<const ArrayNode*>(ptr)->size());
});

// ADT

TVM_REGISTER_OBJECT_TYPE(ADTObj);

TVM_REGISTER_GLOBAL("runtime.GetADTTag").set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectRef obj = args[0];
  const auto& adt = Downcast<ADT>(obj);
  *rv = static_cast<int64_t>(adt.tag());
});

TVM_REGISTER_GLOBAL("runtime.GetADTSize").set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectRef obj = args[0];
  const auto& adt = Downcast<ADT>(obj);
  *rv = static_cast<int64_t>(adt.size());
});

TVM_REGISTER_GLOBAL("runtime.GetADTFields").set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectRef obj = args[0];
  int idx = args[1];
  const auto& adt = Downcast<ADT>(obj);
  ICHECK_LT(idx, adt.size());
  *rv = adt[idx];
});

TVM_REGISTER_GLOBAL("runtime.Tuple").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::vector<ObjectRef> fields;
  for (auto i = 0; i < args.size(); ++i) {
    fields.push_back(args[i]);
  }
  *rv = ADT::Tuple(fields);
});

TVM_REGISTER_GLOBAL("runtime.ADT").set_body([](TVMArgs args, TVMRetValue* rv) {
  int itag = args[0];
  size_t tag = static_cast<size_t>(itag);
  std::vector<ObjectRef> fields;
  for (int i = 1; i < args.size(); i++) {
    fields.push_back(args[i]);
  }
  *rv = ADT(tag, fields);
});

// String
TVM_REGISTER_OBJECT_TYPE(StringObj);

TVM_REGISTER_GLOBAL("runtime.String").set_body_typed([](std::string str) {
  return String(std::move(str));
});

TVM_REGISTER_GLOBAL("runtime.GetFFIString").set_body_typed([](String str) {
  return std::string(str);
});

// Map
TVM_REGISTER_OBJECT_TYPE(MapNode);

TVM_REGISTER_GLOBAL("runtime.Map").set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_EQ(args.size() % 2, 0);
  std::unordered_map<ObjectRef, ObjectRef, ObjectPtrHash, ObjectPtrEqual> data;
  for (int i = 0; i < args.num_args; i += 2) {
    ObjectRef k =
        String::CanConvertFrom(args[i]) ? args[i].operator String() : args[i].operator ObjectRef();
    ObjectRef v = args[i + 1];
    data.emplace(std::move(k), std::move(v));
  }
  *ret = Map<ObjectRef, ObjectRef>(std::move(data));
});

TVM_REGISTER_GLOBAL("runtime.MapSize").set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_EQ(args[0].type_code(), kTVMObjectHandle);
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  ICHECK(ptr->IsInstance<MapNode>());
  auto* n = static_cast<const MapNode*>(ptr);
  *ret = static_cast<int64_t>(n->size());
});

TVM_REGISTER_GLOBAL("runtime.MapGetItem").set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_EQ(args[0].type_code(), kTVMObjectHandle);
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  ICHECK(ptr->IsInstance<MapNode>());

  auto* n = static_cast<const MapNode*>(ptr);
  auto it = n->find(String::CanConvertFrom(args[1]) ? args[1].operator String()
                                                    : args[1].operator ObjectRef());
  ICHECK(it != n->end()) << "cannot find the corresponding key in the Map";
  *ret = (*it).second;
});

TVM_REGISTER_GLOBAL("runtime.MapCount").set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_EQ(args[0].type_code(), kTVMObjectHandle);
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  ICHECK(ptr->IsInstance<MapNode>());
  const MapNode* n = static_cast<const MapNode*>(ptr);
  int64_t cnt = n->count(String::CanConvertFrom(args[1]) ? args[1].operator String()
                                                         : args[1].operator ObjectRef());
  *ret = cnt;
});

TVM_REGISTER_GLOBAL("runtime.MapItems").set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_EQ(args[0].type_code(), kTVMObjectHandle);
  Object* ptr = static_cast<Object*>(args[0].value().v_handle);
  auto* n = static_cast<const MapNode*>(ptr);
  Array<ObjectRef> rkvs;
  for (const auto& kv : *n) {
    if (kv.first->IsInstance<StringObj>()) {
      rkvs.push_back(Downcast<String>(kv.first));
    } else {
      rkvs.push_back(kv.first);
    }
    rkvs.push_back(kv.second);
  }
  *ret = std::move(rkvs);
});

// Closure
TVM_REGISTER_OBJECT_TYPE(ClosureObj);

// ShapeTuple
TVM_REGISTER_OBJECT_TYPE(ShapeTupleObj);

TVM_REGISTER_GLOBAL("runtime.ShapeTuple").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::vector<ShapeTuple::index_type> shape;
  for (int i = 0; i < args.size(); i++) {
    shape.push_back(args[i]);
  }
  *rv = ShapeTuple(shape);
});

TVM_REGISTER_GLOBAL("runtime.GetShapeTupleSize").set_body_typed([](ShapeTuple shape) {
  return static_cast<int64_t>(shape.size());
});

TVM_REGISTER_GLOBAL("runtime.GetShapeTupleElem").set_body_typed([](ShapeTuple shape, int idx) {
  ICHECK_LT(idx, shape.size());
  return shape[idx];
});

}  // namespace runtime
}  // namespace tvm
