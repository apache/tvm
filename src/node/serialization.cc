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
#include <tvm/ir/attrs.h>
#include <tvm/node/reflection.h>
#include <tvm/node/serialization.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cctype>
#include <map>
#include <string>

#include "../runtime/object_internal.h"
#include "../support/base64.h"

namespace tvm {

inline std::string Type2String(const DataType& t) { return runtime::DLDataType2String(t); }

inline DataType String2Type(std::string s) { return DataType(runtime::String2DLDataType(s)); }

inline std::string Base64Decode(std::string s) {
  dmlc::MemoryStringStream mstrm(&s);
  support::Base64InStream b64strm(&mstrm);
  std::string output;
  b64strm.InitPosition();
  dmlc::Stream* strm = &b64strm;
  strm->Read(&output);
  return output;
}

inline std::string Base64Encode(std::string s) {
  std::string blob;
  dmlc::MemoryStringStream mstrm(&blob);
  support::Base64OutStream b64strm(&mstrm);
  dmlc::Stream* strm = &b64strm;
  strm->Write(s);
  b64strm.Finish();
  return blob;
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
    ICHECK_EQ(tensor_index_.size(), tensor_list_.size());
    tensor_index_[ptr] = tensor_list_.size();
    tensor_list_.push_back(ptr);
  }

  void Visit(const char* key, ObjectRef* value) final {
    MakeIndex(const_cast<Object*>(value->get()));
  }

  void MakeNodeIndex(Object* node) {
    if (node == nullptr) return;
    ICHECK(node->IsInstance<Object>());

    if (node_index_.count(node)) {
      return;
    }
    ICHECK_EQ(node_index_.size(), node_list_.size());
    node_index_[node] = node_list_.size();
    node_list_.push_back(node);
  }

  // make index of all the children of node
  void MakeIndex(Object* node) {
    if (node == nullptr) return;
    ICHECK(node->IsInstance<Object>());

    if (node_index_.count(node)) {
      return;
    }
    MakeNodeIndex(node);
    if (node->IsInstance<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      for (const auto& sp : *n) {
        MakeIndex(const_cast<Object*>(sp.get()));
      }
    } else if (node->IsInstance<MapNode>()) {
      MapNode* n = static_cast<MapNode*>(node);
      bool is_str_map = std::all_of(n->begin(), n->end(), [](const auto& v) {
        return v.first->template IsInstance<StringObj>();
      });
      if (is_str_map) {
        for (const auto& kv : *n) {
          MakeIndex(const_cast<Object*>(kv.second.get()));
        }
      } else {
        for (const auto& kv : *n) {
          MakeIndex(const_cast<Object*>(kv.first.get()));
          MakeIndex(const_cast<Object*>(kv.second.get()));
        }
      }
    } else if (node->IsInstance<relay::LetNode>()) {
      auto pre_visit = [this](const relay::LetNode* op) {
        MakeNodeIndex(const_cast<Object*>(static_cast<const Object*>(op)));
        MakeIndex(const_cast<Object*>(static_cast<const Object*>(op->var.get())));
        MakeIndex(const_cast<Object*>(static_cast<const Object*>(op->value.get())));
        MakeIndex(const_cast<Object*>(static_cast<const Object*>(op->span.get())));
        MakeIndex(const_cast<Object*>(static_cast<const Object*>(op->checked_type_.get())));
        if (!op->body.as<relay::LetNode>()) {
          MakeIndex(const_cast<Object*>(static_cast<const Object*>(op->body.get())));
        }
      };
      auto post_visit = [](const relay::LetNode* op) {};
      if (!reflection_->GetReprBytes(node, nullptr)) {
        relay::ExpandANormalForm(static_cast<relay::LetNode*>(node), pre_visit, post_visit);
      }
    } else {
      // if the node already have repr bytes, no need to visit Attrs.
      if (!reflection_->GetReprBytes(node, nullptr)) {
        reflection_->VisitAttrs(node, this);
      }
    }
  }
};

// use map so attributes are ordered.
using AttrMap = std::map<std::string, std::string>;

/*! \brief Node structure for json format. */
struct JSONNode {
  /*! \brief The type of key of the object. */
  std::string type_key;
  /*! \brief The str repr representation. */
  std::string repr_bytes;
  /*! \brief the attributes */
  AttrMap attrs;
  /*! \brief keys of a map. */
  std::vector<std::string> keys;
  /*! \brief values of a map or array. */
  std::vector<size_t> data;
  /*!
   * \brief field member dependency.
   * NOTE: This is an auxiliary data structure for loading, and it won't be serialized to json.
   */
  std::vector<size_t> fields;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("type_key", type_key);
    if (repr_bytes.size() != 0) {
      // choose to use str representation or base64, based on whether
      // the byte representation is printable.
      if (std::all_of(repr_bytes.begin(), repr_bytes.end(),
                      [](char ch) { return std::isprint(ch); })) {
        writer->WriteObjectKeyValue("repr_str", repr_bytes);
      } else {
        writer->WriteObjectKeyValue("repr_b64", Base64Encode(repr_bytes));
      }
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

  void Load(dmlc::JSONReader* reader) {
    attrs.clear();
    data.clear();
    repr_bytes.clear();
    type_key.clear();
    std::string repr_b64, repr_str;
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareOptionalField("type_key", &type_key);
    helper.DeclareOptionalField("repr_b64", &repr_b64);
    helper.DeclareOptionalField("repr_str", &repr_str);
    helper.DeclareOptionalField("attrs", &attrs);
    helper.DeclareOptionalField("keys", &keys);
    helper.DeclareOptionalField("data", &data);
    helper.ReadAllFields(reader);

    if (repr_str.size() != 0) {
      ICHECK_EQ(repr_b64.size(), 0U);
      repr_bytes = std::move(repr_str);
    } else if (repr_b64.size() != 0) {
      repr_bytes = Base64Decode(repr_b64);
    }
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
    // Save 17 decimal digits for type <double> to avoid precision loss during loading JSON
    s.precision(17);
    s << (*value);
    node_->attrs[key] = s.str();
  }
  void Visit(const char* key, int64_t* value) final { node_->attrs[key] = std::to_string(*value); }
  void Visit(const char* key, uint64_t* value) final { node_->attrs[key] = std::to_string(*value); }
  void Visit(const char* key, int* value) final { node_->attrs[key] = std::to_string(*value); }
  void Visit(const char* key, bool* value) final { node_->attrs[key] = std::to_string(*value); }
  void Visit(const char* key, std::string* value) final { node_->attrs[key] = *value; }
  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "not allowed to serialize a pointer";
  }
  void Visit(const char* key, DataType* value) final { node_->attrs[key] = Type2String(*value); }

  void Visit(const char* key, runtime::NDArray* value) final {
    node_->attrs[key] =
        std::to_string(tensor_index_->at(const_cast<DLTensor*>((*value).operator->())));
  }

  void Visit(const char* key, ObjectRef* value) final {
    node_->attrs[key] = std::to_string(node_index_->at(const_cast<Object*>(value->get())));
  }

  // Get the node
  void Get(Object* node) {
    if (node == nullptr) {
      node_->type_key.clear();
      return;
    }
    node_->type_key = node->GetTypeKey();
    // do not need to print additional things once we have repr bytes.
    if (reflection_->GetReprBytes(node, &(node_->repr_bytes))) return;

    // populates the fields.
    node_->attrs.clear();
    node_->data.clear();

    if (node->IsInstance<ArrayNode>()) {
      ArrayNode* n = static_cast<ArrayNode*>(node);
      for (size_t i = 0; i < n->size(); ++i) {
        node_->data.push_back(node_index_->at(const_cast<Object*>(n->at(i).get())));
      }
    } else if (node->IsInstance<MapNode>()) {
      MapNode* n = static_cast<MapNode*>(node);
      bool is_str_map = std::all_of(n->begin(), n->end(), [](const auto& v) {
        return v.first->template IsInstance<StringObj>();
      });
      if (is_str_map) {
        for (const auto& kv : *n) {
          node_->keys.push_back(Downcast<String>(kv.first));
          node_->data.push_back(node_index_->at(const_cast<Object*>(kv.second.get())));
        }
      } else {
        for (const auto& kv : *n) {
          node_->data.push_back(node_index_->at(const_cast<Object*>(kv.first.get())));
          node_->data.push_back(node_index_->at(const_cast<Object*>(kv.second.get())));
        }
      }
    } else {
      // recursively index normal object.
      reflection_->VisitAttrs(node, this);
    }
  }
};

class FieldDependencyFinder : public AttrVisitor {
 public:
  JSONNode* jnode_;
  ReflectionVTable* reflection_ = ReflectionVTable::Global();

  std::string GetValue(const char* key) const {
    auto it = jnode_->attrs.find(key);
    if (it == jnode_->attrs.end()) {
      LOG(FATAL) << "JSONReader: cannot find field " << key;
    }
    return it->second;
  }
  template <typename T>
  void ParseValue(const char* key, T* value) const {
    std::istringstream is(GetValue(key));
    is >> *value;
    if (is.fail()) {
      LOG(FATAL) << "Wrong value format for field " << key;
    }
  }
  void Visit(const char* key, double* value) final {}
  void Visit(const char* key, int64_t* value) final {}
  void Visit(const char* key, uint64_t* value) final {}
  void Visit(const char* key, int* value) final {}
  void Visit(const char* key, bool* value) final {}
  void Visit(const char* key, std::string* value) final {}
  void Visit(const char* key, void** value) final {}
  void Visit(const char* key, DataType* value) final {}
  void Visit(const char* key, runtime::NDArray* value) final {}
  void Visit(const char* key, ObjectRef* value) final {
    size_t index;
    ParseValue(key, &index);
    jnode_->fields.push_back(index);
  }
  void Find(Object* node, JSONNode* jnode) {
    // Skip None
    if (node == nullptr) {
      return;
    }
    // Skip the objects that have their own string repr
    if (jnode->repr_bytes.length() > 0 || reflection_->GetReprBytes(node, nullptr)) {
      return;
    }
    // Skip containers
    if (jnode->type_key == ArrayNode::_type_key || jnode->type_key == MapNode::_type_key) {
      return;
    }
    jnode_ = jnode;
    reflection_->VisitAttrs(node, this);
  }
};

// Helper class to set the attributes of a node
// from given json node.
class JSONAttrSetter : public AttrVisitor {
 public:
  const std::vector<ObjectPtr<Object>>* node_list_;
  const std::vector<runtime::NDArray>* tensor_list_;
  JSONNode* jnode_;

  ReflectionVTable* reflection_ = ReflectionVTable::Global();

  std::string GetValue(const char* key) const {
    auto it = jnode_->attrs.find(key);
    if (it == jnode_->attrs.end()) {
      LOG(FATAL) << "JSONReader: cannot find field " << key;
    }
    return it->second;
  }

  void ParseDouble(const char* key, double* value) const {
    std::istringstream is(GetValue(key));
    if (is.str() == "inf") {
      *value = std::numeric_limits<double>::infinity();
    } else if (is.str() == "-inf") {
      *value = -std::numeric_limits<double>::infinity();
    } else {
      is >> *value;
      if (is.fail()) {
        LOG(FATAL) << "Wrong value format for field " << key;
      }
    }
  }

  template <typename T>
  void ParseValue(const char* key, T* value) const {
    std::istringstream is(GetValue(key));
    is >> *value;
    if (is.fail()) {
      LOG(FATAL) << "Wrong value format for field " << key;
    }
  }
  void Visit(const char* key, double* value) final { ParseDouble(key, value); }
  void Visit(const char* key, int64_t* value) final { ParseValue(key, value); }
  void Visit(const char* key, uint64_t* value) final { ParseValue(key, value); }
  void Visit(const char* key, int* value) final { ParseValue(key, value); }
  void Visit(const char* key, bool* value) final { ParseValue(key, value); }
  void Visit(const char* key, std::string* value) final { *value = GetValue(key); }
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
    ICHECK_LE(index, tensor_list_->size());
    *value = tensor_list_->at(index);
  }
  void Visit(const char* key, ObjectRef* value) final {
    size_t index;
    ParseValue(key, &index);
    ICHECK_LE(index, node_list_->size());
    *value = ObjectRef(node_list_->at(index));
  }
  // set node to be current JSONNode
  void Set(ObjectPtr<Object>* node, JSONNode* jnode) {
    // Skip None
    if (node->get() == nullptr) {
      return;
    }
    // Skip the objects that have their own string repr
    if (jnode->repr_bytes.length() > 0 || reflection_->GetReprBytes(node->get(), nullptr)) {
      return;
    }
    // handling Array
    if (jnode->type_key == ArrayNode::_type_key) {
      std::vector<ObjectRef> container;
      for (auto index : jnode->data) {
        container.push_back(ObjectRef(node_list_->at(index)));
      }
      Array<ObjectRef> array(container);
      *node = runtime::ObjectInternal::MoveObjectPtr(&array);
      return;
    }
    // handling Map
    if (jnode->type_key == MapNode::_type_key) {
      std::unordered_map<ObjectRef, ObjectRef, ObjectHash, ObjectEqual> container;
      if (jnode->keys.empty()) {
        ICHECK_EQ(jnode->data.size() % 2, 0U);
        for (size_t i = 0; i < jnode->data.size(); i += 2) {
          container[ObjectRef(node_list_->at(jnode->data[i]))] =
              ObjectRef(node_list_->at(jnode->data[i + 1]));
        }
      } else {
        ICHECK_EQ(jnode->data.size(), jnode->keys.size());
        for (size_t i = 0; i < jnode->data.size(); ++i) {
          container[String(jnode->keys[i])] = ObjectRef(node_list_->at(jnode->data[i]));
        }
      }
      Map<ObjectRef, ObjectRef> map(container);
      *node = runtime::ObjectInternal::MoveObjectPtr(&map);
      return;
    }
    jnode_ = jnode;
    reflection_->VisitAttrs(node->get(), this);
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

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("root", root);
    writer->WriteObjectKeyValue("nodes", nodes);
    writer->WriteObjectKeyValue("b64ndarrays", b64ndarrays);
    if (attrs.size() != 0) {
      writer->WriteObjectKeyValue("attrs", attrs);
    }
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
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

  std::vector<size_t> TopoSort() const {
    size_t n_nodes = nodes.size();
    std::vector<size_t> topo_order;
    std::vector<size_t> in_degree(n_nodes, 0);
    for (const JSONNode& jnode : nodes) {
      for (size_t i : jnode.data) {
        ++in_degree[i];
      }
      for (size_t i : jnode.fields) {
        ++in_degree[i];
      }
    }
    for (size_t i = 0; i < n_nodes; ++i) {
      if (in_degree[i] == 0) {
        topo_order.push_back(i);
      }
    }
    for (size_t p = 0; p < topo_order.size(); ++p) {
      const JSONNode& jnode = nodes[topo_order[p]];
      for (size_t i : jnode.data) {
        if (--in_degree[i] == 0) {
          topo_order.push_back(i);
        }
      }
      for (size_t i : jnode.fields) {
        if (--in_degree[i] == 0) {
          topo_order.push_back(i);
        }
      }
    }
    ICHECK_EQ(topo_order.size(), n_nodes) << "Cyclic reference detected in JSON file";
    std::reverse(std::begin(topo_order), std::end(topo_order));
    return topo_order;
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
  ReflectionVTable* reflection = ReflectionVTable::Global();
  JSONGraph jgraph;
  {
    // load in json graph.
    std::istringstream is(json_str);
    dmlc::JSONReader reader(&is);
    jgraph.Load(&reader);
  }
  size_t n_nodes = jgraph.nodes.size();
  std::vector<runtime::NDArray> tensors;
  {
    // load in tensors
    for (const std::string& blob : jgraph.b64ndarrays) {
      dmlc::MemoryStringStream mstrm(const_cast<std::string*>(&blob));
      support::Base64InStream b64strm(&mstrm);
      b64strm.InitPosition();
      runtime::NDArray temp;
      ICHECK(temp.Load(&b64strm));
      tensors.emplace_back(std::move(temp));
    }
  }
  // Pass 1: create all non-container objects
  std::vector<ObjectPtr<Object>> nodes(n_nodes, nullptr);
  for (size_t i = 0; i < n_nodes; ++i) {
    const JSONNode& jnode = jgraph.nodes[i];
    if (jnode.type_key.length() != 0) {
      nodes[i] = reflection->CreateInitObject(jnode.type_key, jnode.repr_bytes);
    }
  }
  // Pass 2: figure out all field dependency
  {
    FieldDependencyFinder dep_finder;
    for (size_t i = 0; i < n_nodes; ++i) {
      dep_finder.Find(nodes[i].get(), &jgraph.nodes[i]);
    }
  }
  // Pass 3: topo sort
  std::vector<size_t> topo_order = jgraph.TopoSort();
  // Pass 4: set all values
  {
    JSONAttrSetter setter;
    setter.node_list_ = &nodes;
    setter.tensor_list_ = &tensors;
    for (size_t i : topo_order) {
      setter.Set(&nodes[i], &jgraph.nodes[i]);
    }
  }
  return ObjectRef(nodes.at(jgraph.root));
}

TVM_REGISTER_GLOBAL("node.SaveJSON").set_body_typed(SaveJSON);

TVM_REGISTER_GLOBAL("node.LoadJSON").set_body_typed(LoadJSON);
}  // namespace tvm
