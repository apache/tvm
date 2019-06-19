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
 * \file reflection.cc
 * \brief Utilities to save/load/construct TVM objects
 */
#include <tvm/base.h>
#include <tvm/expr.h>
#include <tvm/attrs.h>
#include <tvm/node/container.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/json.h>
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/common/base64.h>
#include <string>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::tvm::NodeFactoryReg);
}  // namespace dmlc

namespace tvm {

::dmlc::Registry<NodeFactoryReg>* NodeFactoryReg::Registry() {
  return ::dmlc::Registry<NodeFactoryReg>::Get();
}

inline Type String2Type(std::string s) {
  return TVMType2Type(runtime::String2TVMType(s));
}

class JSONAttrSetter : public AttrVisitor {
 public:
  const std::vector<NodePtr<Node> >* node_list_;
  const std::vector<runtime::NDArray>* tensor_list_;
  const std::vector<Object>* vm_obj_list_;

  JSONNode* node_;

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
  void Visit(const char* key, Type* value) final {
    std::string stype = GetValue(key);
    *value = String2Type(stype);
  }
  void Visit(const char* key, NodeRef* value) final {
    size_t index;
    ParseValue(key, &index);
    CHECK_LE(index, node_list_->size());
    value->node_ = node_list_->at(index);
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    size_t index;
    ParseValue(key, &index);
    CHECK_LE(index, tensor_list_->size());
    *value = tensor_list_->at(index);
  }
  void Visit(const char* key, Object* value) final {
    size_t index;
    ParseValue(key, &index);
    CHECK_LE(index, vm_obj_list_->size());
    *value = vm_obj_list_->at(index);
  }
  // set node to be current JSONNode
  void Set(Node* node) {
    if (node == nullptr) return;
    if (node->is_type<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      n->data.clear();
      for (size_t index : node_->data) {
        n->data.push_back(node_list_->at(index));
      }
    } else if (node->is_type<MapNode>()) {
      MapNode* n = static_cast<MapNode*>(node);
      CHECK_EQ(node_->data.size() % 2, 0U);
      for (size_t i = 0; i < node_->data.size(); i += 2) {
        n->data[node_list_->at(node_->data[i])]
            = node_list_->at(node_->data[i + 1]);
      }
    } else if (node->is_type<StrMapNode>()) {
      StrMapNode* n = static_cast<StrMapNode*>(node);
      CHECK_EQ(node_->data.size(), node_->keys.size());
      for (size_t i = 0; i < node_->data.size(); ++i) {
        n->data[node_->keys[i]]
            = node_list_->at(node_->data[i]);
      }
    } else {
      node->VisitAttrs(this);
    }
  }
};

std::string SaveJSON(const NodeRef& n) {
  auto jgraph = JSONGraph::Create(n);
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  jgraph.Save(&writer);
  return os.str();
}

NodePtr<Node> LoadJSON_(std::string json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JSONGraph jgraph;
  // load in json graph.
  jgraph.Load(&reader);
  std::vector<NodePtr<Node> > nodes;
  std::vector<runtime::NDArray> tensors;
  // load in tensors
  for (const std::string& blob : jgraph.b64ndarrays) {
    dmlc::MemoryStringStream mstrm(const_cast<std::string*>(&blob));
    common::Base64InStream b64strm(&mstrm);
    b64strm.InitPosition();
    runtime::NDArray temp;
    CHECK(temp.Load(&b64strm));
    tensors.emplace_back(temp);
  }
  // node 0 is always null
  nodes.reserve(jgraph.nodes.size());
  for (const JSONNode& jnode : jgraph.nodes) {
    if (jnode.type_key.length() != 0) {
      auto* f = dmlc::Registry<NodeFactoryReg>::Find(jnode.type_key);
      CHECK(f != nullptr)
          << "Node type \'" << jnode.type_key << "\' is not registered in TVM";
      nodes.emplace_back(f->fcreator(jnode.global_key));
    } else {
      nodes.emplace_back(NodePtr<Node>());
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
  return nodes.at(jgraph.root);
}

class NodeAttrSetter : public AttrVisitor {
 public:
  std::string type_key;
  std::unordered_map<std::string, runtime::TVMArgValue> attrs;

  void Visit(const char* key, double* value) final {
    *value = GetAttr(key).operator double();
  }
  void Visit(const char* key, int64_t* value) final {
    *value = GetAttr(key).operator int64_t();
  }
  void Visit(const char* key, uint64_t* value) final {
    *value = GetAttr(key).operator uint64_t();
  }
  void Visit(const char* key, int* value) final {
    *value = GetAttr(key).operator int();
  }
  void Visit(const char* key, bool* value) final {
    *value = GetAttr(key).operator bool();
  }
  void Visit(const char* key, std::string* value) final {
    *value = GetAttr(key).operator std::string();
  }
  void Visit(const char* key, void** value) final {
    *value = GetAttr(key).operator void*();
  }
  void Visit(const char* key, Type* value) final {
    *value = GetAttr(key).operator Type();
  }
  void Visit(const char* key, NodeRef* value) final {
    *value = GetAttr(key).operator NodeRef();
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    *value = GetAttr(key).operator runtime::NDArray();
  }
  void Visit(const char* key, Object* value) final {
    *value = GetAttr(key).operator Object();
  }

 private:
  runtime::TVMArgValue GetAttr(const char* key) {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      LOG(FATAL) << type_key << ": require field " << key;
    }
    runtime::TVMArgValue v = it->second;
    attrs.erase(it);
    return v;
  }
};


void InitNodeByPackedArgs(Node* n, const TVMArgs& args) {
  NodeAttrSetter setter;
  setter.type_key = n->type_key();
  CHECK_EQ(args.size() % 2, 0);
  for (int i = 0; i < args.size(); i += 2) {
    setter.attrs.emplace(args[i].operator std::string(),
                         args[i + 1]);
  }
  n->VisitAttrs(&setter);
  if (setter.attrs.size() != 0) {
    std::ostringstream os;
    os << setter.type_key << " does not contain field ";
    for (const auto &kv : setter.attrs) {
      os << " " << kv.first;
    }
    LOG(FATAL) << os.str();
  }
}

// API function to make node.
// args format:
//   key1, value1, ..., key_n, value_n
void MakeNode(const TVMArgs& args, TVMRetValue* rv) {
  std::string type_key = args[0];
  std::string empty_str;
  auto* f = dmlc::Registry<NodeFactoryReg>::Find(type_key);
  CHECK(f != nullptr)
      << "Node type \'" << type_key << "\' is not registered in TVM";
  TVMArgs kwargs(args.values + 1, args.type_codes + 1, args.size() - 1);
  CHECK(f->fglobal_key == nullptr)
      << "Cannot make node type \'" << type_key << "\' with global_key.";
  NodePtr<Node> n = f->fcreator(empty_str);
  if (n->derived_from<BaseAttrsNode>()) {
    static_cast<BaseAttrsNode*>(n.get())->InitByPackedArgs(kwargs);
  } else {
    InitNodeByPackedArgs(n.get(), kwargs);
  }
  *rv = NodeRef(n);
}

TVM_REGISTER_GLOBAL("make._Node")
.set_body(MakeNode);
}  // namespace tvm
