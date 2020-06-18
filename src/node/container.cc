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
#include <tvm/runtime/container.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include "../support/str_escape.h"

namespace tvm {

// SEQualReduce traits for runtime containers.
struct StringObjTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const runtime::StringObj* key,
                          SHashReducer hash_reduce) {
    hash_reduce->SHashReduceHashedValue(
        runtime::String::HashBytes(key->data, key->size));
  }

  static bool SEqualReduce(const runtime::StringObj* lhs,
                           const runtime::StringObj* rhs,
                           SEqualReducer equal) {
    if (lhs == rhs) return true;
    if (lhs->size != rhs->size) return false;
    if (lhs->data == rhs->data) return true;
    return std::memcmp(lhs->data, rhs->data, lhs->size) == 0;
  }
};

struct RefToObjectPtr : public ObjectRef {
  static ObjectPtr<Object> Get(const ObjectRef& ref) {
    return GetDataPtr<Object>(ref);
  }
};

TVM_REGISTER_REFLECTION_VTABLE(runtime::StringObj, StringObjTrait)
.set_creator([](const std::string& bytes) {
  return RefToObjectPtr::Get(runtime::String(bytes));
})
.set_repr_bytes([](const Object* n) -> std::string {
  return GetRef<runtime::String>(
      static_cast<const runtime::StringObj*>(n)).operator std::string();
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<runtime::StringObj>([](const ObjectRef& node, ReprPrinter* p) {
  auto* op = static_cast<const runtime::StringObj*>(node.get());
  p->stream << '"' << support::StrEscape(op->data, op->size) << '"';
});


struct ADTObjTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const runtime::ADTObj* key,
                          SHashReducer hash_reduce) {
    hash_reduce(key->tag);
    hash_reduce(static_cast<uint64_t>(key->size));
    for (uint32_t i = 0; i < key->size; ++i) {
      hash_reduce((*key)[i]);
    }
  }

  static bool SEqualReduce(const runtime::ADTObj* lhs,
                           const runtime::ADTObj* rhs,
                           SEqualReducer equal) {
    if (lhs == rhs) return true;
    if (lhs->tag != rhs->tag) return false;
    if (lhs->size != rhs->size) return false;

    for (uint32_t i = 0; i < lhs->size; ++i) {
      if (!equal((*lhs)[i], (*rhs)[i])) return false;
    }
    return true;
  }
};

TVM_REGISTER_REFLECTION_VTABLE(runtime::ADTObj, ADTObjTrait);


struct NDArrayContainerTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const runtime::NDArray::Container* key,
                          SHashReducer hash_reduce) {
    CHECK_EQ(key->dl_tensor.ctx.device_type, kDLCPU) << "can only compare CPU tensor";
    CHECK(runtime::IsContiguous(key->dl_tensor))
        << "Can only hash contiguous tensor";
    hash_reduce(runtime::DataType(key->dl_tensor.dtype));
    hash_reduce(key->dl_tensor.ndim);
    for (int i = 0; i < key->dl_tensor.ndim; ++i) {
      hash_reduce(key->dl_tensor.shape[i]);
    }
    hash_reduce->SHashReduceHashedValue(
        runtime::String::HashBytes(
            static_cast<const char*>(key->dl_tensor.data),
            runtime::GetDataSize(key->dl_tensor)));
  }

  static bool SEqualReduce(const runtime::NDArray::Container* lhs,
                           const runtime::NDArray::Container* rhs,
                           SEqualReducer equal) {
    if (lhs == rhs) return true;

    auto ldt = lhs->dl_tensor.dtype;
    auto rdt = rhs->dl_tensor.dtype;
    CHECK_EQ(lhs->dl_tensor.ctx.device_type, kDLCPU) << "can only compare CPU tensor";
    CHECK_EQ(rhs->dl_tensor.ctx.device_type, kDLCPU) << "can only compare CPU tensor";
    CHECK(runtime::IsContiguous(lhs->dl_tensor))
        << "Can only compare contiguous tensor";
    CHECK(runtime::IsContiguous(rhs->dl_tensor))
        << "Can only compare contiguous tensor";

    if (lhs->dl_tensor.ndim != rhs->dl_tensor.ndim) return false;
    for (int i = 0; i < lhs->dl_tensor.ndim; ++i) {
      if (!equal(lhs->dl_tensor.shape[i], rhs->dl_tensor.shape[i])) return false;
    }
    if (ldt.code == rdt.code && ldt.lanes == rdt.lanes && ldt.bits == rdt.bits) {
      size_t data_size = runtime::GetDataSize(lhs->dl_tensor);
      return std::memcmp(lhs->dl_tensor.data, rhs->dl_tensor.data, data_size) == 0;
    } else {
      return false;
    }
  }
};

TVM_REGISTER_REFLECTION_VTABLE(runtime::NDArray::Container, NDArrayContainerTrait);


struct ArrayNodeTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const ArrayNode* key,
                          SHashReducer hash_reduce) {
    hash_reduce(static_cast<uint64_t>(key->data.size()));
    for (size_t i = 0; i < key->data.size(); ++i) {
      hash_reduce(key->data[i]);
    }
  }

  static bool SEqualReduce(const ArrayNode* lhs,
                           const ArrayNode* rhs,
                           SEqualReducer equal) {
    if (lhs->data.size() != rhs->data.size()) return false;
    for (size_t i = 0; i < lhs->data.size(); ++i) {
      if (!equal(lhs->data[i], rhs->data[i])) return false;
    }
    return true;
  }
};

TVM_REGISTER_OBJECT_TYPE(ArrayNode);
TVM_REGISTER_REFLECTION_VTABLE(ArrayNode, ArrayNodeTrait)
.set_creator([](const std::string&) -> ObjectPtr<Object> {
    return ::tvm::runtime::make_object<ArrayNode>();
  });


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


struct MapNodeTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const MapNode* key,
                          SHashReducer hash_reduce) {
    // SHash's var handling depends on the determinism of traversal.
    // NOTE: only book-keep the mapped hash keys.
    // This resolves common use cases where we want to store
    // Map<Var, Value> where Var is defined in the function
    // parameters.
    using KV = std::pair<size_t, ObjectRef>;
    std::vector<KV> temp;
    for (const auto& kv : key->data) {
      size_t hashed_value;
      if (hash_reduce->LookupHashedValue(kv.first, &hashed_value)) {
        temp.emplace_back(hashed_value, kv.second);
      }
    }
    // sort by the hash key of the keys.
    std::sort(temp.begin(), temp.end(), [](const KV& lhs, const KV& rhs) {
      return lhs.first < rhs.first;
    });
    // add size to the hash
    hash_reduce(static_cast<uint64_t>(key->data.size()));
    // hash the content
    for (size_t i = 0; i < temp.size();) {
      size_t k = i + 1;
      for (; k < temp.size() && temp[k].first == temp[i].first; ++k) {}
      // ties are rare, but we need to skip them to make the hash determinsitic
      if (k == i + 1) {
        hash_reduce->SHashReduceHashedValue(temp[i].first);
        hash_reduce(temp[i].second);
      }
      i = k;
    }
  }

  static bool SEqualReduce(const MapNode* lhs,
                           const MapNode* rhs,
                           SEqualReducer equal) {
    if (rhs->data.size() != lhs->data.size()) return false;
    for (const auto& kv : lhs->data) {
      // Only allow equal checking if the keys are already mapped
      // This resolves common use cases where we want to store
      // Map<Var, Value> where Var is defined in the function
      // parameters.
      ObjectRef rhs_key = equal->MapLhsToRhs(kv.first);
      if (!rhs_key.defined()) return false;
      auto it = rhs->data.find(rhs_key);
      if (it == rhs->data.end()) return false;
      if (!equal(kv.second, it->second)) return false;
    }
    return true;
  }
};

TVM_REGISTER_OBJECT_TYPE(MapNode);
TVM_REGISTER_REFLECTION_VTABLE(MapNode, MapNodeTrait)
.set_creator([](const std::string&) -> ObjectPtr<Object> {
    return ::tvm::runtime::make_object<MapNode>();
  });


struct StrMapNodeTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const StrMapNode* key,
                          SHashReducer hash_reduce) {
    // NOTE: only book-keep the mapped hash keys.
    // This resolves common use cases where we want to store
    // Map<Var, Value> where Var is defined in the function
    // parameters.
    using KV = std::pair<std::string, ObjectRef>;
    std::vector<KV> temp(key->data.begin(), key->data.end());
    // sort by the hash key of the keys.
    std::sort(temp.begin(), temp.end(), [](const KV& lhs, const KV& rhs) {
      return lhs.first < rhs.first;
    });
    // NOTE: we won't have ties
    // add size to the hash after sorting.
    hash_reduce(static_cast<uint64_t>(key->data.size()));
    // hash the content
    for (size_t i = 0; i < temp.size(); ++i) {
      hash_reduce(temp[i].first);
      hash_reduce(temp[i].second);
    }
  }

  static bool SEqualReduce(const StrMapNode* lhs,
                           const StrMapNode* rhs,
                           SEqualReducer equal) {
    if (rhs->data.size() != lhs->data.size()) return false;
    for (const auto& kv : lhs->data) {
      auto it = rhs->data.find(kv.first);
      if (it == rhs->data.end()) return false;
      if (!equal(kv.second, it->second)) return false;
    }
    return true;
  }
};

TVM_REGISTER_OBJECT_TYPE(StrMapNode);
TVM_REGISTER_REFLECTION_VTABLE(StrMapNode, StrMapNodeTrait)
.set_creator([](const std::string&) -> ObjectPtr<Object> {
    return ::tvm::runtime::make_object<StrMapNode>();
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
