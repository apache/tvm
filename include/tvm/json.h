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
 * \file json.h
 * \brief A representation of JSON
 */

#ifndef TVM_JSON_H_
#define TVM_JSON_H_

#include <dmlc/memory_io.h>
#include <tvm/common/base64.h>

#include <map>
#include <unordered_map>
#include <vector>
#include <utility>
#include <string>

namespace tvm {

// use map so attributes are ordered.
using AttrMap = std::map<std::string, std::string>;

using runtime::Object;
using runtime::ObjectCell;

inline std::string Type2String(const Type& t) {
  return runtime::TVMType2String(Type2TVMType(t));
}

// indexer to index all the ndoes
class NodeIndexer : public AttrVisitor {
 public:
  std::unordered_map<Node*, size_t> node_index{{nullptr, 0}};
  std::vector<Node*> node_list{nullptr};
  std::unordered_map<DLTensor*, size_t> tensor_index;
  std::vector<DLTensor*> tensor_list;
  std::unordered_map<ObjectCell*, size_t> vm_obj_index;
  std::vector<ObjectCell*> vm_obj_list;

  void Visit(const char* key, double* value) final {}
  void Visit(const char* key, int64_t* value) final {}
  void Visit(const char* key, uint64_t* value) final {}
  void Visit(const char* key, int* value) final {}
  void Visit(const char* key, bool* value) final {}
  void Visit(const char* key, std::string* value) final {}
  void Visit(const char* key, void** value) final {}
  void Visit(const char* key, Type* value) final {}
  void Visit(const char* key, NodeRef* value) final {
    MakeIndex(value->node_.get());
  }

  void Visit(const char* key, runtime::NDArray* value) final {
    DLTensor* ptr = const_cast<DLTensor*>((*value).operator->());
    if (tensor_index.count(ptr)) return;
    CHECK_EQ(tensor_index.size(), tensor_list.size());
    tensor_index[ptr] = tensor_list.size();
    tensor_list.push_back(ptr);
  }

  void Visit(const char* key, Object* value) final {
    ObjectCell* ptr = value->ptr_.get();
    if (vm_obj_index.count(ptr)) return;
    CHECK_EQ(vm_obj_index.size(), vm_obj_list.size());
    vm_obj_index[ptr] = vm_obj_list.size();
    vm_obj_list.push_back(ptr);
  }

  // make index of all the children of node
  void MakeIndex(Node* node) {
    if (node == nullptr) return;
    if (node_index.count(node)) return;
    CHECK_EQ(node_index.size(), node_list.size());
    node_index[node] = node_list.size();
    node_list.push_back(node);

    if (node->is_type<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      for (const auto& sp : n->data) {
        MakeIndex(sp.get());
      }
    } else if (node->is_type<MapNode>()) {
      MapNode* n = static_cast<MapNode*>(node);
      for (const auto& kv : n->data) {
        MakeIndex(kv.first.get());
        MakeIndex(kv.second.get());
      }
    } else if (node->is_type<StrMapNode>()) {
      StrMapNode* n = static_cast<StrMapNode*>(node);
      for (const auto& kv : n->data) {
        MakeIndex(kv.second.get());
      }
    } else {
      node->VisitAttrs(this);
    }
  }
};

// A Node structure for JSON node.
struct JSONNode {
  // The type key of the data
  std::string type_key;
  // The global key for global object
  std::string global_key;
  // the attributes
  AttrMap attrs;
  // container keys
  std::vector<std::string> keys;
  // container data
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

class JSONAttrGetter : public AttrVisitor {
 public:
  const std::unordered_map<Node*, size_t>* node_index_;
  const std::unordered_map<DLTensor*, size_t>* tensor_index_;
  const std::unordered_map<ObjectCell*, size_t>* vm_obj_index_;
  JSONNode* node_;

  void Visit(const char* key, double* value) final {
    node_->attrs[key] = std::to_string(*value);
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
  void Visit(const char* key, Type* value) final {
    node_->attrs[key] = Type2String(*value);
  }
  void Visit(const char* key, NodeRef* value) final {
    node_->attrs[key] = std::to_string(
        node_index_->at(value->node_.get()));
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    node_->attrs[key] = std::to_string(
        tensor_index_->at(const_cast<DLTensor*>((*value).operator->())));
  }
  void Visit(const char* key, Object* value) final {
    node_->attrs[key] = std::to_string(
        vm_obj_index_->at(value->ptr_.get()));
  }
  // Get the node
  void Get(Node* node) {
    if (node == nullptr) {
      node_->type_key.clear();
      return;
    }
    node_->type_key = node->type_key();
    // sepcially handle global object
    auto* f = dmlc::Registry<NodeFactoryReg>::Find(node_->type_key);
    CHECK(f != nullptr)
        << "Node type \'" << node_->type_key << "\' is not registered in TVM";
    if (f->fglobal_key != nullptr) {
      node_->global_key = f->fglobal_key(node);
      return;
    }
    node_->attrs.clear();
    node_->data.clear();
    if (node->is_type<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      for (size_t i = 0; i < n->data.size(); ++i) {
        node_->data.push_back(
            node_index_->at(n->data[i].get()));
      }
    } else if (node->is_type<MapNode>()) {
      MapNode* n = static_cast<MapNode*>(node);
      for (const auto& kv : n->data) {
        node_->data.push_back(
            node_index_->at(kv.first.get()));
        node_->data.push_back(
            node_index_->at(kv.second.get()));
      }
    } else if (node->is_type<StrMapNode>()) {
      StrMapNode* n = static_cast<StrMapNode*>(node);
      for (const auto& kv : n->data) {
        node_->keys.push_back(kv.first);
        node_->data.push_back(
            node_index_->at(kv.second.get()));
      }
    } else {
      // do not need to recover content of global singleton object
      // they are registered via the environment
      auto* f = dmlc::Registry<NodeFactoryReg>::Find(node->type_key());
      if (f != nullptr && f->fglobal_key != nullptr) return;
      // recursively index normal object.
      node->VisitAttrs(this);
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

  static JSONGraph Create(const NodeRef& root) {
    JSONGraph g;
    NodeIndexer indexer;
    indexer.MakeIndex(root.node_.get());
    JSONAttrGetter getter;
    getter.node_index_ = &indexer.node_index;
    getter.tensor_index_ = &indexer.tensor_index;
    for (Node* n : indexer.node_list) {
      JSONNode jnode;
      getter.node_ = &jnode;
      getter.Get(n);
      g.nodes.emplace_back(std::move(jnode));
    }
    g.attrs["tvm_version"] = TVM_VERSION;
    g.root = indexer.node_index.at(root.node_.get());
    // serialize tensor
    for (DLTensor* tensor : indexer.tensor_list) {
      std::string blob;
      dmlc::MemoryStringStream mstrm(&blob);
      common::Base64OutStream b64strm(&mstrm);
      runtime::SaveDLTensor(&b64strm, tensor);
      b64strm.Finish();
      g.b64ndarrays.emplace_back(std::move(blob));
    }
    return g;
  }
};

}  // namespace tvm
#endif  // TVM_JSON_H_
