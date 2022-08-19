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

#include <tvm/script/printer/traced_object.h>
#include <tvm/script/printer/visit_traced.h>

namespace tvm {
namespace script {
namespace printer {

void PostOrderVisitTracedImpl(const ObjectRef& object, const ObjectPath& path,
                              const std::function<bool(const ObjectRef&)>& node_predicate,
                              const std::function<void(const TracedObject<ObjectRef>&)>& callback);

struct ObjAttrVisitor : public AttrVisitor {
  ObjAttrVisitor(const ObjectPath& path, const std::function<bool(const ObjectRef&)> node_predicate,
                 const std::function<void(const TracedObject<ObjectRef>&)>& callback)
      : path(path), node_predicate(node_predicate), callback(callback) {}

  const ObjectPath& path;
  const std::function<bool(const ObjectRef&)> node_predicate;
  const std::function<void(const TracedObject<ObjectRef>&)>& callback;

  void Visit(const char* key, double* value) final {}
  void Visit(const char* key, int64_t* value) final {}
  void Visit(const char* key, uint64_t* value) final {}
  void Visit(const char* key, int* value) final {}
  void Visit(const char* key, bool* value) final {}
  void Visit(const char* key, void** value) final {}
  void Visit(const char* key, DataType* value) final {}
  void Visit(const char* key, std::string* value) final {}
  void Visit(const char* key, runtime::NDArray* value) final {}
  void Visit(const char* key, ObjectRef* value) final {
    PostOrderVisitTracedImpl(*value, path->Attr(key), node_predicate, callback);
  }
};

void PostOrderVisitTracedImpl(const ObjectRef& object, const ObjectPath& path,
                              const std::function<bool(const ObjectRef&)>& node_predicate,
                              const std::function<void(const TracedObject<ObjectRef>&)>& callback) {
  if (!object.defined()) {
    return;
  }

  if (object->IsInstance<ArrayNode>()) {
    const ArrayNode* node = static_cast<const ArrayNode*>(object.get());
    for (size_t i = 0; i < node->size(); ++i) {
      PostOrderVisitTracedImpl(node->at(i), path->ArrayIndex(i), node_predicate, callback);
    }
  } else if (object->IsInstance<MapNode>()) {
    const MapNode* node = static_cast<const MapNode*>(object.get());
    for (auto kv : *node) {
      PostOrderVisitTracedImpl(kv.second, path->MapValue(kv.first), node_predicate, callback);
    }
  } else {
    if (!node_predicate(object)) {
      return;
    }

    ObjAttrVisitor visitor(path, node_predicate, callback);
    ReflectionVTable::Global()->VisitAttrs(const_cast<Object*>(object.get()), &visitor);

    callback(MakeTraced(object, path));
  }
}

void PostOrderVisitTraced(const TracedObject<ObjectRef>& object,
                          const std::function<bool(const ObjectRef&)>& node_predicate,
                          const std::function<void(const TracedObject<ObjectRef>&)>& callback) {
  PostOrderVisitTracedImpl(object.Get(), object.GetPath(), node_predicate, callback);
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
