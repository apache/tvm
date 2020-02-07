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
 *  Expose container API to frontend.
 * \file src/node/container.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>

namespace tvm {

TVM_REGISTER_GLOBAL("node.Array")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    std::vector<ObjectRef> data;
    for (int i = 0; i < args.size(); ++i) {
      if (args[i].type_code() != kTVMNullptr) {
        data.push_back(args[i].operator ObjectRef());
      } else {
        data.push_back(ObjectRef(nullptr));
      }
    }
    auto node = make_object<ArrayNode>();
    node->data = std::move(data);
    *ret = Array<ObjectRef>(node);
  });

TVM_REGISTER_GLOBAL("node.ArrayGetItem")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    int64_t i = args[1];
    CHECK_EQ(args[0].type_code(), kTVMObjectHandle);
    Object* ptr = static_cast<Object*>(args[0].value().v_handle);
    CHECK(ptr->IsInstance<ArrayNode>());
    auto* n = static_cast<const ArrayNode*>(ptr);
    CHECK_LT(static_cast<size_t>(i), n->data.size())
        << "out of bound of array";
    *ret = n->data[static_cast<size_t>(i)];
  });

TVM_REGISTER_GLOBAL("node.ArraySize")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK_EQ(args[0].type_code(), kTVMObjectHandle);
    Object* ptr = static_cast<Object*>(args[0].value().v_handle);
    CHECK(ptr->IsInstance<ArrayNode>());
    *ret = static_cast<int64_t>(
        static_cast<const ArrayNode*>(ptr)->data.size());
  });

TVM_REGISTER_GLOBAL("node.Map")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK_EQ(args.size() % 2, 0);
    if (args.size() != 0 && args[0].type_code() == kTVMStr) {
      // StrMap
      StrMapNode::ContainerType data;
      for (int i = 0; i < args.num_args; i += 2) {
        CHECK(args[i].type_code() == kTVMStr)
            << "key of str map need to be str";
        CHECK(args[i + 1].IsObjectRef<ObjectRef>())
            << "value of the map to be NodeRef";
        data.emplace(std::make_pair(args[i].operator std::string(),
                                    args[i + 1].operator ObjectRef()));
      }
      auto node = make_object<StrMapNode>();
      node->data = std::move(data);
      *ret = Map<ObjectRef, ObjectRef>(node);
    } else {
      // Container node.
      MapNode::ContainerType data;
      for (int i = 0; i < args.num_args; i += 2) {
        CHECK(args[i].IsObjectRef<ObjectRef>())
            << "key of str map need to be object";
        CHECK(args[i + 1].IsObjectRef<ObjectRef>())
            << "value of map to be NodeRef";
        data.emplace(std::make_pair(args[i].operator ObjectRef(),
                                    args[i + 1].operator ObjectRef()));
      }
      auto node = make_object<MapNode>();
      node->data = std::move(data);
      *ret = Map<ObjectRef, ObjectRef>(node);
    }
  });


TVM_REGISTER_GLOBAL("node.MapSize")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK_EQ(args[0].type_code(), kTVMObjectHandle);
    Object* ptr = static_cast<Object*>(args[0].value().v_handle);
    if (ptr->IsInstance<MapNode>()) {
      auto* n = static_cast<const MapNode*>(ptr);
      *ret = static_cast<int64_t>(n->data.size());
    } else {
      CHECK(ptr->IsInstance<StrMapNode>());
      auto* n = static_cast<const StrMapNode*>(ptr);
      *ret = static_cast<int64_t>(n->data.size());
    }
  });

TVM_REGISTER_GLOBAL("node.MapGetItem")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK_EQ(args[0].type_code(), kTVMObjectHandle);
    Object* ptr = static_cast<Object*>(args[0].value().v_handle);

    if (ptr->IsInstance<MapNode>()) {
      CHECK(args[1].type_code() == kTVMObjectHandle);
      auto* n = static_cast<const MapNode*>(ptr);
      auto it = n->data.find(args[1].operator ObjectRef());
      CHECK(it != n->data.end())
          << "cannot find the corresponding key in the Map";
      *ret = (*it).second;
    } else {
      CHECK(ptr->IsInstance<StrMapNode>());
      auto* n = static_cast<const StrMapNode*>(ptr);
      auto it = n->data.find(args[1].operator std::string());
      CHECK(it != n->data.end())
          << "cannot find the corresponding key in the Map";
      *ret = (*it).second;
    }
  });

TVM_REGISTER_GLOBAL("node.MapCount")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK_EQ(args[0].type_code(), kTVMObjectHandle);
    Object* ptr = static_cast<Object*>(args[0].value().v_handle);

    if (ptr->IsInstance<MapNode>()) {
      auto* n = static_cast<const MapNode*>(ptr);
    CHECK_EQ(args[0].type_code(), kTVMObjectHandle);
      *ret = static_cast<int64_t>(
          n->data.count(args[1].operator ObjectRef()));
    } else {
      CHECK(ptr->IsInstance<StrMapNode>());
      auto* n = static_cast<const StrMapNode*>(ptr);
      *ret = static_cast<int64_t>(
          n->data.count(args[1].operator std::string()));
    }
  });

TVM_REGISTER_GLOBAL("node.MapItems")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    CHECK_EQ(args[0].type_code(), kTVMObjectHandle);
    Object* ptr = static_cast<Object*>(args[0].value().v_handle);

    if (ptr->IsInstance<MapNode>()) {
      auto* n = static_cast<const MapNode*>(ptr);
      auto rkvs = make_object<ArrayNode>();
      for (const auto& kv : n->data) {
        rkvs->data.push_back(kv.first);
        rkvs->data.push_back(kv.second);
      }
      *ret = Array<ObjectRef>(rkvs);
    } else {
      auto* n = static_cast<const StrMapNode*>(ptr);
      auto rkvs = make_object<ArrayNode>();
      for (const auto& kv : n->data) {
        rkvs->data.push_back(tir::StringImmNode::make(kv.first));
        rkvs->data.push_back(kv.second);
      }
      *ret = Array<ObjectRef>(rkvs);
    }
  });
}  // namespace tvm
