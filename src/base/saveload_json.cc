/*!
 *  Copyright (c) 2016 by Contributors
 * \file saveload_json.cc
 * \brief Utilities to save/load TVM objects.
 */
#include <tvm/base.h>
#include <tvm/container.h>
#include <dmlc/json.h>
#include <string>
#include "./common.h"

namespace tvm {

// indexer to index all the ndoes
class NodeIndexer : public AttrVisitor {
 public:
  std::unordered_map<Node*, size_t> node_index{{nullptr, 0}};
  std::vector<Node*> node_list{nullptr};

  void Visit(const char* key, double* value) final {}
  void Visit(const char* key, int64_t* value) final {}
  void Visit(const char* key, uint64_t* value) final {}
  void Visit(const char* key, int* value) final {}
  void Visit(const char* key, bool* value) final {}
  void Visit(const char* key, std::string* value) final {}
  void Visit(const char* key, Type* value) final {}
  void Visit(const char* key, NodeRef* value) final {
    MakeIndex(value->node_.get());
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
    } else {
      node->VisitAttrs(this);
    }
  }
};

// use map so attributes are ordered.
using AttrMap = std::map<std::string, std::string>;

// A Node structure for JSON node.
struct JSONNode {
  // The type key of the data
  std::string type_key;
  // the attributes
  AttrMap attrs;
  // container data
  std::vector<size_t> data;

  void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("type_key", type_key);
    if (attrs.size() != 0) {
      writer->WriteObjectKeyValue("attrs", attrs);
    }
    if (data.size() != 0) {
      writer->WriteObjectKeyValue("data", data);
    }
    writer->EndObject();
  }

  void Load(dmlc::JSONReader *reader) {
    attrs.clear();
    data.clear();
    type_key.clear();
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareOptionalField("type_key", &type_key);
    helper.DeclareOptionalField("attrs", &attrs);
    helper.DeclareOptionalField("data", &data);
    helper.ReadAllFields(reader);
  }
};

class JSONAttrGetter : public AttrVisitor {
 public:
  const std::unordered_map<Node*, size_t>* node_index_;
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
  void Visit(const char* key, Type* value) final {
    node_->attrs[key] = Type2String(*value);
  }
  void Visit(const char* key, NodeRef* value) final {
    node_->attrs[key] = std::to_string(
        node_index_->at(value->node_.get()));
  }
  // Get the node
  void Get(Node* node) {
    if (node == nullptr) {
      node_->type_key.clear();
      return;
    }
    node_->type_key = node->type_key();
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
      std::vector<std::pair<size_t, size_t> > elems;
      for (const auto& kv : n->data) {
        node_->data.push_back(
            node_index_->at(kv.first.get()));
        node_->data.push_back(
            node_index_->at(kv.second.get()));
      }
    } else {
      node->VisitAttrs(this);
    }
  }
};

class JSONAttrSetter : public AttrVisitor {
 public:
  const std::vector<std::shared_ptr<Node> >* node_list_;
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
  void Visit(const char* key, Type* value) final {
    std::string stype = GetValue(key);
    *value = String2Type(stype);
  }
  void Visit(const char* key, NodeRef* value) final {
    size_t index;
    ParseValue(key, &index);
    value->node_ = node_list_->at(index);
  }

  // Get the node
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
    } else {
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
  // global attributes
  AttrMap attrs;

  void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("root", root);
    writer->WriteObjectKeyValue("nodes", nodes);
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
    helper.DeclareOptionalField("attrs", &attrs);
    helper.ReadAllFields(reader);
  }

  static JSONGraph Create(const NodeRef& root) {
    JSONGraph g;
    NodeIndexer indexer;
    indexer.MakeIndex(root.node_.get());
    JSONAttrGetter getter;
    getter.node_index_ = &indexer.node_index;
    for (Node* n : indexer.node_list) {
      JSONNode jnode;
      getter.node_ = &jnode;
      getter.Get(n);
      g.nodes.emplace_back(std::move(jnode));
    }
    g.attrs["tvm_version"] = "0.1.0";
    g.root = indexer.node_index.at(root.node_.get());
    return g;
  }
};

std::string SaveJSON(const NodeRef& n) {
  auto jgraph = JSONGraph::Create(n);
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  jgraph.Save(&writer);
  return os.str();
}

std::shared_ptr<Node> LoadJSON_(std::string json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JSONGraph jgraph;
  // load in json graph.
  jgraph.Load(&reader);
  std::vector<std::shared_ptr<Node> > nodes;
  // node 0 is always null
  nodes.reserve(jgraph.nodes.size());
  for (const JSONNode& jnode : jgraph.nodes) {
    if (jnode.type_key.length() != 0) {
      auto* f = dmlc::Registry<NodeFactoryReg>::Find(jnode.type_key);
      CHECK(f != nullptr)
          << "Node type \'" << jnode.type_key << "\' is not registered in TVM";
      nodes.emplace_back(f->body());
    } else {
      nodes.emplace_back(std::shared_ptr<Node>());
    }
  }
  CHECK_EQ(nodes.size(), jgraph.nodes.size());
  JSONAttrSetter setter;
  setter.node_list_ = &nodes;

  for (size_t i = 0; i < nodes.size(); ++i) {
    setter.node_ = &jgraph.nodes[i];
    setter.Set(nodes[i].get());
  }
  return nodes.at(jgraph.root);
}

}  // namespace tvm
