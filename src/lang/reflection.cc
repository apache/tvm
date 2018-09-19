/*!
 *  Copyright (c) 2016 by Contributors
 * \file reflection.cc
 * \brief Utilities to save/load/construct TVM objects
 */
#include <tvm/base.h>
#include <tvm/expr.h>
#include <tvm/attrs.h>
#include <tvm/node/container.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/ndarray.h>
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <string>
#include "../common/base64.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::tvm::NodeFactoryReg);
}  // namespace dmlc

namespace tvm {

::dmlc::Registry<NodeFactoryReg>* NodeFactoryReg::Registry() {
  return ::dmlc::Registry<NodeFactoryReg>::Get();
}

inline std::string Type2String(const Type& t) {
  if (t.code()  ==Type::Handle) return "handle";
  std::ostringstream os;
  os << t;
  return os.str();
}


inline Type String2Type(std::string s) {
  std::istringstream is(s);
  halideir_type_code_t code = Type::Int;
  if (s.substr(0, 3) == "int") {
    code = Type::Int; s = s.substr(3);
  } else if (s.substr(0, 4) == "uint") {
    code = Type::UInt; s = s.substr(4);
  } else if (s.substr(0, 5) == "float") {
    code = Type::Float; s = s.substr(5);
  } else if (s.substr(0, 5) == "float") {
    code = Type::Float; s = s.substr(5);
  } else if (s == "handle") {
    return Handle();
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  int bits = 32, lanes = 1;
  if (sscanf(s.c_str(), "%dx%d", &bits, &lanes) == 0) {
    LOG(FATAL) << "unknown type " << s;
  }
  return Type(code, bits, lanes);
}


// indexer to index all the ndoes
class NodeIndexer : public AttrVisitor {
 public:
  std::unordered_map<Node*, size_t> node_index{{nullptr, 0}};
  std::vector<Node*> node_list{nullptr};
  std::unordered_map<DLTensor*, size_t> tensor_index;
  std::vector<DLTensor*> tensor_list;

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

// use map so attributes are ordered.
using AttrMap = std::map<std::string, std::string>;

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
  // Get the node
  void Get(Node* node) {
    if (node == nullptr) {
      node_->type_key.clear();
      return;
    }
    node_->type_key = node->type_key();
    // sepcially handle global object
    auto* f = dmlc::Registry<NodeFactoryReg>::Find(node_->type_key);
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

class JSONAttrSetter : public AttrVisitor {
 public:
  const std::vector<NodePtr<Node> >* node_list_;
  const std::vector<runtime::NDArray>* tensor_list_;
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
