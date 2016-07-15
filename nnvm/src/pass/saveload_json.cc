/*!
 *  Copyright (c) 2016 by Contributors
 * \file saveload_json.cc
 * \brief Save and load graph to/from JSON file.
 */
#include <nnvm/pass.h>
#include <dmlc/json.h>
#include <algorithm>

namespace dmlc {
namespace json {
// overload handler for shared ptr
template<>
struct Handler<std::shared_ptr<const any> > {
  inline static void Write(JSONWriter *writer, const std::shared_ptr<const any> &data) {
    writer->Write(*data);
  }
  inline static void Read(JSONReader *reader, std::shared_ptr<const any> *data) {
    any v;
    reader->Read(&v);
    *data = std::make_shared<any>(std::move(v));
  }
};
}  // namespace json
}  // namespace dmlc

namespace nnvm {
namespace pass {

// auxiliary node structure for serialization.
struct JSONNode {
  // the node entry structure in serialized format
  typedef std::pair<uint32_t, uint32_t> Entry;
  // pointer to the graph node
  NodePtr node;
  // inputs
  std::vector<Entry> inputs;
  // control flow dependencies
  std::vector<uint32_t> control_deps;

  // function to save JSON node.
  void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    if (node->op != nullptr) {
      writer->WriteObjectKeyValue("op", node->op->name);
      writer->WriteObjectKeyValue("attr", node->attrs.dict);
    } else {
      std::string json_null = "null";
      writer->WriteObjectKeyValue("op", json_null);
    }
    writer->WriteObjectKeyValue("name", node->attrs.name);
    writer->WriteObjectKeyValue("inputs", inputs);
    writer->WriteObjectKeyValue("control_deps", control_deps);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader *reader) {
    node = std::move(Node::Create());
    control_deps.clear();
    dmlc::JSONObjectReadHelper helper;
    std::string op_type_str;
    helper.DeclareField("op", &op_type_str);
    helper.DeclareField("name", &(node->attrs.name));
    helper.DeclareField("inputs", &inputs);
    helper.DeclareOptionalField("attr", &(node->attrs.dict));
    helper.DeclareOptionalField("control_deps", &control_deps);
    // backward compatible code with mxnet graph.
    int backward_source_id;
    std::unordered_map<std::string, std::string> param;
    helper.DeclareOptionalField("param", &param);
    helper.DeclareOptionalField("backward_source_id", &backward_source_id);
    node->attrs.dict.insert(param.begin(), param.end());
    helper.ReadAllFields(reader);

    if (op_type_str != "null") {
      try {
        node->op = Op::Get(op_type_str);
      } catch (const dmlc::Error &err) {
        std::ostringstream os;
        os << "Failed loading Op " << node->attrs.name
           << " of type " << op_type_str << ": " << err.what();
        throw dmlc::Error(os.str());
      }
    } else {
      node->op = nullptr;
    }
  }
};

// graph structure to help read/save JSON.
struct JSONGraph {
  std::vector<JSONNode> nodes;
  std::vector<uint32_t> arg_nodes;
  std::vector<JSONNode::Entry> heads;
  std::unordered_map<std::string, std::shared_ptr<const any> > attrs;

  void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("nodes", nodes);
    writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
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
    helper.DeclareOptionalField("attrs", &attrs);
    helper.ReadAllFields(reader);
  }
};

// Load a graph from JSON file.
Graph LoadJSON(const Graph& src) {
  CHECK_NE(src.attrs.count("json"), 0)
      << "Load JSON require json to be presented.";
  const std::string &json_str =
      nnvm::get<std::string>(*src.attrs.at("json"));
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JSONGraph jgraph;
  // load in json graph.
  jgraph.Load(&reader);
  // connects the nodes
  for (JSONNode &n : jgraph.nodes) {
    n.node->inputs.reserve(n.inputs.size());
    for (const JSONNode::Entry &e : n.inputs) {
      n.node->inputs.emplace_back(
          NodeEntry{jgraph.nodes[e.first].node, e.second});
    }
    n.node->control_deps.reserve(n.control_deps.size());
    for (uint32_t nid : n.control_deps) {
      n.node->control_deps.push_back(jgraph.nodes[nid].node);
    }
  }
  // consistent check
  for (uint32_t nid : jgraph.arg_nodes) {
    CHECK(jgraph.nodes[nid].node->is_variable());
  }

  // return the graph
  Graph ret;
  ret.attrs = std::move(jgraph.attrs);
  ret.outputs.reserve(jgraph.heads.size());
  for (const JSONNode::Entry &e : jgraph.heads) {
    ret.outputs.emplace_back(
        NodeEntry{jgraph.nodes[e.first].node, e.second});
  }
  return ret;
}

// save a graph to json
Graph SaveJSON(const Graph& src) {
  JSONGraph jgraph;
  std::unordered_map<Node*, uint32_t> node2index;
  DFSVisit(src.outputs, [&node2index, &jgraph](const NodePtr& n) {
      uint32_t nid = static_cast<uint32_t>(jgraph.nodes.size());
      node2index[n.get()] = nid;
      if (n->is_variable()) {
        jgraph.arg_nodes.push_back(nid);
      }
      JSONNode jnode;
      jnode.node = n;
      jnode.inputs.reserve(n->inputs.size());
      for (const NodeEntry& e : n->inputs) {
        jnode.inputs.emplace_back(
            std::make_pair(node2index.at(e.node.get()), e.index));
      }
      for (const NodePtr& c : n->control_deps) {
        jnode.control_deps.push_back(node2index.at(c.get()));
      }
      jgraph.nodes.emplace_back(std::move(jnode));
    });

  for (const NodeEntry& e : src.outputs) {
    jgraph.heads.push_back(std::make_pair(node2index.at(e.node.get()), e.index));
  }

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

}  // namespace pass
}  // namespace nnvm
