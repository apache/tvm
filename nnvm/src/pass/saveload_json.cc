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
 *  Copyright (c) 2016 by Contributors
 * \file saveload_json.cc
 * \brief Save and load graph to/from JSON file.
 */
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <dmlc/json.h>
#include <algorithm>

namespace dmlc {
namespace json {
// overload handler for shared ptr
template<>
struct Handler<std::shared_ptr<any> > {
  inline static void Write(JSONWriter *writer, const std::shared_ptr<any> &data) {
    writer->Write(*data);
  }
  inline static void Read(JSONReader *reader, std::shared_ptr<any> *data) {
    any v;
    reader->Read(&v);
    *data = std::make_shared<any>(std::move(v));
  }
};
}  // namespace json
}  // namespace dmlc

namespace nnvm {
namespace pass {
namespace {

// JSONNode represents an nnvm::Node in JSON
struct JSONNode;
// JSONGraph represents an nnvm::Graph or nnvm::Symbol in JSON
struct JSONGraph;

// auxiliary node structure for serialization.
struct JSONNode {
  // the node entry structure in serialized format
  struct Entry {
    uint32_t node_id;
    uint32_t index;
    uint32_t version;
    Entry() = default;
    Entry(uint32_t node_id, uint32_t index, uint32_t version):
      node_id(node_id), index(index), version(version) {
    }
    void Save(dmlc::JSONWriter *writer) const {
      writer->BeginArray(false);
      writer->WriteArrayItem(node_id);
      writer->WriteArrayItem(index);
      writer->WriteArrayItem(version);
      writer->EndArray();
    }
    void Load(dmlc::JSONReader *reader) {
      reader->BeginArray();
      CHECK(reader->NextArrayItem()) << "invalid json format";
      reader->Read(&node_id);
      CHECK(reader->NextArrayItem()) << "invalid json format";
      reader->Read(&index);
      if (reader->NextArrayItem()) {
        reader->Read(&version);
        CHECK(!reader->NextArrayItem()) << "invalid json format";
      } else {
        version = 0;
      }
    }
  };

  // pointer to the graph node
  NodePtr node;
  // inputs
  std::vector<Entry> inputs;
  // control flow dependencies
  std::vector<uint32_t> control_deps;
  // subgraphs
  std::vector<JSONGraph> subgraphs;

  // function to save JSON node.
  void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    if (node->op() != nullptr) {
      writer->WriteObjectKeyValue("op", node->op()->name);
    } else {
      std::string json_null = "null";
      writer->WriteObjectKeyValue("op", json_null);
    }
    writer->WriteObjectKeyValue("name", node->attrs.name);
    if (node->attrs.dict.size() != 0) {
      // write attributes in order;
      std::map<std::string, std::string> dict(
          node->attrs.dict.begin(), node->attrs.dict.end());
      writer->WriteObjectKeyValue("attrs", dict);
    }
    writer->WriteObjectKeyValue("inputs", inputs);
    if (control_deps.size() != 0) {
      writer->WriteObjectKeyValue("control_deps", control_deps);
    }
    if (subgraphs.size() != 0) {
      writer->WriteObjectKeyValue("subgraphs", subgraphs);
    }
    writer->EndObject();
  }

  void Load(dmlc::JSONReader *reader) {
    node = Node::Create();
    control_deps.clear();
    dmlc::JSONObjectReadHelper helper;
    std::string op_type_str;
    helper.DeclareField("op", &op_type_str);
    helper.DeclareField("name", &(node->attrs.name));
    helper.DeclareField("inputs", &inputs);
    helper.DeclareOptionalField("attrs", &(node->attrs.dict));
    helper.DeclareOptionalField("attr", &(node->attrs.dict));
    helper.DeclareOptionalField("control_deps", &control_deps);
    helper.DeclareOptionalField("subgraphs", &subgraphs);
    // backward compatible code with mxnet graph.
    int backward_source_id;
    std::unordered_map<std::string, std::string> param;
    helper.DeclareOptionalField("param", &param);
    helper.DeclareOptionalField("backward_source_id", &backward_source_id);
    helper.ReadAllFields(reader);
    node->attrs.dict.insert(param.begin(), param.end());

    if (op_type_str != "null") {
      try {
        node->attrs.op = Op::Get(op_type_str);
      } catch (const dmlc::Error &err) {
        std::ostringstream os;
        os << "Failed loading Op " << node->attrs.name
           << " of type " << op_type_str << ": " << err.what();
        throw dmlc::Error(os.str());
      }
    } else {
      node->attrs.op = nullptr;
    }
  }
};

// graph structure to help read/save JSON.
struct JSONGraph {
  std::vector<JSONNode> nodes;
  std::vector<uint32_t> arg_nodes;
  std::vector<uint32_t> node_row_ptr;
  std::vector<JSONNode::Entry> heads;
  std::unordered_map<std::string, std::shared_ptr<any> > attrs;

  void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("nodes", nodes);
    writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
    writer->WriteObjectKeyValue("node_row_ptr", node_row_ptr);
    writer->WriteObjectKeyValue("heads", heads);
    if (attrs.size() != 0) {
      writer->WriteObjectKeyValue("attrs", attrs);
    }
    writer->EndObject();
  }

  void Load(dmlc::JSONReader *reader) {
    attrs.clear();
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("nodes", &nodes);
    helper.DeclareField("arg_nodes", &arg_nodes);
    helper.DeclareField("heads", &heads);
    helper.DeclareOptionalField("node_row_ptr", &node_row_ptr);
    helper.DeclareOptionalField("attrs", &attrs);
    helper.ReadAllFields(reader);
  }
};

void Symbol2JSONGraph(std::shared_ptr<Symbol> src, JSONGraph *jgraph) {
  std::unordered_map<Node*, uint32_t> node2index;
  jgraph->node_row_ptr.push_back(0);
  DFSVisit(src->outputs, [&node2index, jgraph](const NodePtr& n) {
    uint32_t nid = static_cast<uint32_t>(jgraph->nodes.size());
    node2index[n.get()] = nid;
    if (n->is_variable()) {
      jgraph->arg_nodes.push_back(nid);
    }
    JSONNode jnode;
    jnode.node = n;
    jnode.inputs.reserve(n->inputs.size());
    for (const NodeEntry& e : n->inputs) {
      jnode.inputs.emplace_back(node2index.at(e.node.get()), e.index, e.version);
    }
    for (const NodePtr& c : n->control_deps) {
      jnode.control_deps.push_back(node2index.at(c.get()));
    }
    jgraph->node_row_ptr.push_back(jgraph->node_row_ptr.back() + n->num_outputs());
    jgraph->nodes.emplace_back(std::move(jnode));
  });
  for (const NodeEntry& e : src->outputs) {
    jgraph->heads.emplace_back(node2index.at(e.node.get()), e.index, e.version);
  }
  // recursively construct subgraphs
  for (JSONNode &jnode : jgraph->nodes) {
    // construct jnode's subgraphs
    const std::vector<std::shared_ptr<Symbol>> &subgraphs = jnode.node->attrs.subgraphs;
    std::vector<JSONGraph> &jsubgraphs = jnode.subgraphs;
    jsubgraphs.resize(subgraphs.size());
    for (uint32_t i = 0; i < subgraphs.size(); ++i) {
      Symbol2JSONGraph(subgraphs[i], &jsubgraphs[i]);
    }
  }
}

std::shared_ptr<Symbol> JSONGraph2Symbol(const JSONGraph &jgraph, bool no_parse) {
  for (const JSONNode &n : jgraph.nodes) {
    n.node->inputs.reserve(n.inputs.size());
    for (const JSONNode::Entry &e : n.inputs) {
      CHECK(e.node_id < jgraph.nodes.size());
      n.node->inputs.emplace_back(NodeEntry{jgraph.nodes[e.node_id].node, e.index, e.version});
    }
    n.node->control_deps.reserve(n.control_deps.size());
    for (uint32_t nid : n.control_deps) {
      CHECK(nid < jgraph.nodes.size());
      n.node->control_deps.push_back(jgraph.nodes[nid].node);
    }
    for (const JSONGraph &subgraph : n.subgraphs) {
      // The "no_parse" option here, is to be compatible with
      // commit cfd3075e85807dcd8f9534c37e053583dee87524
      // (https://github.com/apache/incubator-mxnet/tree/cfd3075e85807dcd8f9534c37e053583dee87524),
      // where the parsing of main graph is deferred until
      // incubator-mxnet/src/nnvm/legacy_json_util.cc:UpgradeJSON_Parse
      n.node->attrs.subgraphs.push_back(JSONGraph2Symbol(subgraph, false));
    }
    // rebuild attribute parser
    if (!no_parse && n.node->op() != nullptr && n.node->op()->attr_parser != nullptr) {
      n.node->op()->attr_parser(&(n.node->attrs));
    } else if (!no_parse && n.node->is_variable()) {
      n.node->attrs.parsed =
        Symbol::CreateVariable(n.node->attrs.name).outputs[0].node->attrs.parsed;
    }
  }
  // consistency check
  for (uint32_t nid : jgraph.arg_nodes) {
    CHECK(nid < jgraph.nodes.size());
    CHECK(jgraph.nodes[nid].node->is_variable());
  }
  std::shared_ptr<Symbol> symbol = std::make_shared<Symbol>();
  symbol->outputs.reserve(jgraph.heads.size());
  for (const JSONNode::Entry &e : jgraph.heads) {
    CHECK(e.node_id < jgraph.nodes.size());
    symbol->outputs.emplace_back(NodeEntry{jgraph.nodes[e.node_id].node, e.index, e.version});
  }
  return symbol;
}

// Load a graph from JSON file.
Graph LoadJSON(Graph src) {
  CHECK_NE(src.attrs.count("json"), 0U)
      << "Load JSON require json to be presented.";
  const std::string &json_str =
      nnvm::get<std::string>(*src.attrs.at("json"));
  bool no_parse = false;
  if (src.attrs.count("load_json_no_parse")) {
    no_parse = nnvm::get<bool>(*src.attrs.at("load_json_no_parse"));
  }
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JSONGraph jgraph;
  // load in json graph.
  jgraph.Load(&reader);
  std::shared_ptr<Symbol> symbol = JSONGraph2Symbol(jgraph, no_parse);
  // return the graph
  Graph ret;
  ret.attrs = std::move(jgraph.attrs);
  ret.outputs = symbol->outputs;
  return ret;
}

// save a graph to json
Graph SaveJSON(Graph src) {
  std::shared_ptr<Symbol> src_symbol = std::make_shared<Symbol>();
  src_symbol->outputs = src.outputs;
  JSONGraph jgraph;
  Symbol2JSONGraph(src_symbol, &jgraph);
  jgraph.attrs = src.attrs;
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  jgraph.Save(&writer);
  Graph ret;
  ret.attrs["json"] = std::make_shared<any>(os.str());
  return ret;
}

// register pass
NNVM_REGISTER_PASS(LoadJSON)
.describe("Return a new Graph, loaded from src.attrs[\"json\"]")
.set_body(LoadJSON)
.set_change_graph(true)
.depend_graph_attr("json");

NNVM_REGISTER_PASS(SaveJSON)
.describe("Return a new empty Graph. Save graph to ret.attrs[\"json\"]")
.set_body(SaveJSON)
.set_change_graph(true)
.provide_graph_attr("json");


DMLC_JSON_ENABLE_ANY(std::string, str);
DMLC_JSON_ENABLE_ANY(std::vector<int>, list_int);
DMLC_JSON_ENABLE_ANY(std::vector<std::string>, list_str);

}  // namespace
}  // namespace pass
}  // namespace nnvm
