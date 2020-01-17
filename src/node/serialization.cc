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
 * \file node/serialization.cc
 * \brief Utilities to serialize TVM AST/IR objects.
 */
#include <dmlc/json.h>
#include <dmlc/memory_io.h>

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/node/container.h>
#include <tvm/node/reflection.h>
#include <tvm/node/serialization.h>
#include <tvm/ir/attrs.h>

#include <string>
#include <map>

#include "../support/base64.h"

namespace tvm {

inline std::string Type2String(const DataType& t) {
  return runtime::DLDataType2String(t);
}

inline DataType String2Type(std::string s) {
  return DataType(runtime::String2DLDataType(s));
}

// indexer to index all the nodes
class NodeIndexer : public AttrVisitor {
 public:
  std::unordered_map<Object*, size_t> node_index_{{nullptr, 0}};
  std::vector<Object*> node_list_{nullptr};
  std::unordered_map<DLTensor*, size_t> tensor_index_;
  std::vector<DLTensor*> tensor_list_;
  ReflectionVTable* reflection_ = ReflectionVTable::Global();

  void Visit(const char* key, double* value) final {}
  void Visit(const char* key, int64_t* value) final {}
  void Visit(const char* key, uint64_t* value) final {}
  void Visit(const char* key, int* value) final {}
  void Visit(const char* key, bool* value) final {}
  void Visit(const char* key, std::string* value) final {}
  void Visit(const char* key, void** value) final {}
  void Visit(const char* key, DataType* value) final {}

  void Visit(const char* key, runtime::NDArray* value) final {
    DLTensor* ptr = const_cast<DLTensor*>((*value).operator->());
    if (tensor_index_.count(ptr)) return;
    CHECK_EQ(tensor_index_.size(), tensor_list_.size());
    tensor_index_[ptr] = tensor_list_.size();
    tensor_list_.push_back(ptr);
  }

  void Visit(const char* key, ObjectRef* value) final {
    MakeIndex(const_cast<Object*>(value->get()));
  }

  // make index of all the children of node
  void MakeIndex(Object* node) {
    if (node == nullptr) return;
    CHECK(node->IsInstance<Object>());

    if (node_index_.count(node)) return;
    CHECK_EQ(node_index_.size(), node_list_.size());
    node_index_[node] = node_list_.size();
    node_list_.push_back(node);

    if (node->IsInstance<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      for (const auto& sp : n->data) {
        MakeIndex(const_cast<Object*>(sp.get()));
      }
    } else if (node->IsInstance<MapNode>()) {
      MapNode* n = static_cast<MapNode*>(node);
      for (const auto& kv : n->data) {
        MakeIndex(const_cast<Object*>(kv.first.get()));
        MakeIndex(const_cast<Object*>(kv.second.get()));
      }
    } else if (node->IsInstance<StrMapNode>()) {
      StrMapNode* n = static_cast<StrMapNode*>(node);
      for (const auto& kv : n->data) {
        MakeIndex(const_cast<Object*>(kv.second.get()));
      }
    } else {
      reflection_->VisitAttrs(node, this);
    }
  }
};

// use map so attributes are ordered.
using AttrMap = std::map<std::string, std::string>;

/*! \brief Node structure for json format. */
struct JSONNode {
  /*! \brief The type of key of the object. */
  std::string type_key;
  /*! \brief The global key for global object. */
  std::string global_key;
  /*! \brief the attributes */
  AttrMap attrs;
  /*! \brief keys of a map. */
  std::vector<std::string> keys;
  /*! \brief values of a map or array. */
  std::vector<size_t> data;

  void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("type_key", type_key);
    if (global_key.size() != 0) {
      writer->WriteObjectKeyValue("global_key", global_key);
    }
    if (attrs.size() != 0) {
      writer->WriteObjectKeyValue("attrs", attrs);
    }
    if (keys.size() != 0) {
      writer->WriteObjectKeyValue("keys", keys);
    }
    if (data.size() != 0) {
      writer->WriteObjectKeyValue("data", data);
    }
    writer->EndObject();
  }

  void Load(dmlc::JSONReader *reader) {
    attrs.clear();
    data.clear();
    global_key.clear();
    type_key.clear();
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareOptionalField("type_key", &type_key);
    helper.DeclareOptionalField("global_key", &global_key);
    helper.DeclareOptionalField("attrs", &attrs);
    helper.DeclareOptionalField("keys", &keys);
    helper.DeclareOptionalField("data", &data);
    helper.ReadAllFields(reader);
  }
};

// Helper class to populate the json node
// using the existing index.
class JSONAttrGetter : public AttrVisitor {
 public:
  const std::unordered_map<Object*, size_t>* node_index_;
  const std::unordered_map<DLTensor*, size_t>* tensor_index_;
  JSONNode* node_;
  ReflectionVTable* reflection_ = ReflectionVTable::Global();

  void Visit(const char* key, double* value) final {
    std::ostringstream s;
    // Type <double> have approximately 16 decimal digits
    s.precision(16);
    s << (*value);
    node_->attrs[key] = s.str();
  }
  void Visit(const char* key, int64_t* value) final {
    node_->attrs[key] = std::to_string(*value);
  }
  void Visit(const char* key, uint64_t* value) final {
    node_->attrs[key] = std::to_string(*value);
  }
  void Visit(const char* key, int* value) final {
    node_->attrs[key] = std::to_string(*value);
  }
  void Visit(const char* key, bool* value) final {
    node_->attrs[key] = std::to_string(*value);
  }
  void Visit(const char* key, std::string* value) final {
    node_->attrs[key] = *value;
  }
  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "not allowed to serialize a pointer";
  }
  void Visit(const char* key, DataType* value) final {
    node_->attrs[key] = Type2String(*value);
  }

  void Visit(const char* key, runtime::NDArray* value) final {
    node_->attrs[key] = std::to_string(
        tensor_index_->at(const_cast<DLTensor*>((*value).operator->())));
  }

  void Visit(const char* key, ObjectRef* value) final {
    node_->attrs[key] = std::to_string(
        node_index_->at(const_cast<Object*>(value->get())));
  }

  // Get the node
  void Get(Object* node) {
    if (node == nullptr) {
      node_->type_key.clear();
      return;
    }
    node_->type_key = node->GetTypeKey();
    node_->global_key = reflection_->GetGlobalKey(node);
    // No need to recursively visit fields of global singleton
    // They are registered via the environment.
    if (node_->global_key.length() != 0) return;

    // populates the fields.
    node_->attrs.clear();
    node_->data.clear();

    if (node->IsInstance<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      for (size_t i = 0; i < n->data.size(); ++i) {
        node_->data.push_back(
            node_index_->at(const_cast<Object*>(n->data[i].get())));
      }
    } else if (node->IsInstance<MapNode>()) {
      MapNode* n = static_cast<MapNode*>(node);
      for (const auto& kv : n->data) {
        node_->data.push_back(
            node_index_->at(const_cast<Object*>(kv.first.get())));
        node_->data.push_back(
            node_index_->at(const_cast<Object*>(kv.second.get())));
      }
    } else if (node->IsInstance<StrMapNode>()) {
      StrMapNode* n = static_cast<StrMapNode*>(node);
      for (const auto& kv : n->data) {
        node_->keys.push_back(kv.first);
        node_->data.push_back(
            node_index_->at(const_cast<Object*>(kv.second.get())));
      }
    } else {
      // recursively index normal object.
      reflection_->VisitAttrs(node, this);
    }
  }
};

// Helper class to set the attributes of a node
// from given json node.
class JSONAttrSetter : public AttrVisitor {
 public:
  const std::vector<ObjectPtr<Object> >* node_list_;
  const std::vector<runtime::NDArray>* tensor_list_;
  JSONNode* node_;

  ReflectionVTable* reflection_ = ReflectionVTable::Global();

  std::string GetValue(const char* key) const {
    auto it = node_->attrs.find(key);
    if (it == node_->attrs.end()) {
      LOG(FATAL) << "JSONReader: cannot find field " << key;
    }
    return it->second;
  }
  template<typename T>
  void ParseValue(const char* key, T* value) const {
    std::istringstream is(GetValue(key));
    is >> *value;
    if (is.fail()) {
      LOG(FATAL) << "Wrong value format for field " << key;
    }
  }
  void Visit(const char* key, double* value) final {
    ParseValue(key, value);
  }
  void Visit(const char* key, int64_t* value) final {
    ParseValue(key, value);
  }
  void Visit(const char* key, uint64_t* value) final {
    ParseValue(key, value);
  }
  void Visit(const char* key, int* value) final {
    ParseValue(key, value);
  }
  void Visit(const char* key, bool* value) final {
    ParseValue(key, value);
  }
  void Visit(const char* key, std::string* value) final {
    *value = GetValue(key);
  }
  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "not allowed to deserialize a pointer";
  }
  void Visit(const char* key, DataType* value) final {
    std::string stype = GetValue(key);
    *value = String2Type(stype);
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    size_t index;
    ParseValue(key, &index);
    CHECK_LE(index, tensor_list_->size());
    *value = tensor_list_->at(index);
  }
  void Visit(const char* key, ObjectRef* value) final {
    size_t index;
    ParseValue(key, &index);
    CHECK_LE(index, node_list_->size());
    *value = ObjectRef(node_list_->at(index));
  }
  // set node to be current JSONNode
  void Set(Object* node) {
    if (node == nullptr) return;

    if (node->IsInstance<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      n->data.clear();
      for (size_t index : node_->data) {
        n->data.push_back(ObjectRef(node_list_->at(index)));
      }
    } else if (node->IsInstance<MapNode>()) {
      MapNode* n = static_cast<MapNode*>(node);
      CHECK_EQ(node_->data.size() % 2, 0U);
      for (size_t i = 0; i < node_->data.size(); i += 2) {
        n->data[ObjectRef(node_list_->at(node_->data[i]))]
            = ObjectRef(node_list_->at(node_->data[i + 1]));
      }
    } else if (node->IsInstance<StrMapNode>()) {
      StrMapNode* n = static_cast<StrMapNode*>(node);
      CHECK_EQ(node_->data.size(), node_->keys.size());
      for (size_t i = 0; i < node_->data.size(); ++i) {
        n->data[node_->keys[i]]
            = ObjectRef(node_list_->at(node_->data[i]));
      }
    } else {
      reflection_->VisitAttrs(node, this);
    }
  }
};

// json graph structure to store node
struct JSONGraph {
  // the root of the graph
  size_t root;
  // the nodes of the graph
  std::vector<JSONNode> nodes;
  // base64 b64ndarrays of arrays
  std::vector<std::string> b64ndarrays;
  // global attributes
  AttrMap attrs;

  void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("root", root);
    writer->WriteObjectKeyValue("nodes", nodes);
    writer->WriteObjectKeyValue("b64ndarrays", b64ndarrays);
    if (attrs.size() != 0) {
      writer->WriteObjectKeyValue("attrs", attrs);
    }
    writer->EndObject();
  }

  void Load(dmlc::JSONReader *reader) {
    attrs.clear();
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("root", &root);
    helper.DeclareField("nodes", &nodes);
    helper.DeclareOptionalField("b64ndarrays", &b64ndarrays);
    helper.DeclareOptionalField("attrs", &attrs);
    helper.ReadAllFields(reader);
  }

  static JSONGraph Create(const ObjectRef& root) {
    JSONGraph g;
    NodeIndexer indexer;
    indexer.MakeIndex(const_cast<Object*>(root.get()));
    JSONAttrGetter getter;
    getter.node_index_ = &indexer.node_index_;
    getter.tensor_index_ = &indexer.tensor_index_;
    for (Object* n : indexer.node_list_) {
      JSONNode jnode;
      getter.node_ = &jnode;
      getter.Get(n);
      g.nodes.emplace_back(std::move(jnode));
    }
    g.attrs["tvm_version"] = TVM_VERSION;
    g.root = indexer.node_index_.at(const_cast<Object*>(root.get()));
    // serialize tensor
    for (DLTensor* tensor : indexer.tensor_list_) {
      std::string blob;
      dmlc::MemoryStringStream mstrm(&blob);
      support::Base64OutStream b64strm(&mstrm);
      runtime::SaveDLTensor(&b64strm, tensor);
      b64strm.Finish();
      g.b64ndarrays.emplace_back(std::move(blob));
    }
    return g;
  }
};

std::string SaveJSON(const ObjectRef& n) {
  auto jgraph = JSONGraph::Create(n);
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  jgraph.Save(&writer);
  return os.str();
}

ObjectRef LoadJSON(std::string json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JSONGraph jgraph;
  // load in json graph.
  jgraph.Load(&reader);
  std::vector<ObjectPtr<Object> > nodes;
  std::vector<runtime::NDArray> tensors;
  // load in tensors
  for (const std::string& blob : jgraph.b64ndarrays) {
    dmlc::MemoryStringStream mstrm(const_cast<std::string*>(&blob));
    support::Base64InStream b64strm(&mstrm);
    b64strm.InitPosition();
    runtime::NDArray temp;
    CHECK(temp.Load(&b64strm));
    tensors.emplace_back(temp);
  }
  ReflectionVTable* reflection = ReflectionVTable::Global();

  // node 0 is always null
  nodes.reserve(jgraph.nodes.size());

  for (const JSONNode& jnode : jgraph.nodes) {
    if (jnode.type_key.length() != 0) {
      ObjectPtr<Object> node =
          reflection->CreateInitObject(jnode.type_key, jnode.global_key);
      nodes.emplace_back(node);
    } else {
      nodes.emplace_back(ObjectPtr<Object>());
    }
  }
  CHECK_EQ(nodes.size(), jgraph.nodes.size());
  JSONAttrSetter setter;
  setter.node_list_ = &nodes;
  setter.tensor_list_ = &tensors;

  for (size_t i = 0; i < nodes.size(); ++i) {
    setter.node_ = &jgraph.nodes[i];
    // do not need to recover content of global singleton object
    // they are registered via the environment
    if (setter.node_->global_key.length() == 0) {
      setter.Set(nodes[i].get());
    }
  }
  return ObjectRef(nodes.at(jgraph.root));
}
}  // namespace tvm
