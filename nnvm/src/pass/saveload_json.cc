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

// auxiliary node structure for serialization.
struct JSONNode {
  // the node entry structure in serialized format
  struct Entry {
    uint32_t node_id;
    uint32_t index;
    uint32_t version;
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
  // connects the nodes
  for (JSONNode &n : jgraph.nodes) {
    n.node->inputs.reserve(n.inputs.size());
    for (const JSONNode::Entry &e : n.inputs) {
      n.node->inputs.emplace_back(
          NodeEntry{jgraph.nodes[e.node_id].node, e.index, e.version});
    }
    n.node->control_deps.reserve(n.control_deps.size());
    for (uint32_t nid : n.control_deps) {
      n.node->control_deps.push_back(jgraph.nodes[nid].node);
    }
    // rebuild attribute parser
    if (!no_parse && n.node->op() != nullptr &&
        n.node->op()->attr_parser != nullptr) {
      n.node->op()->attr_parser(&(n.node->attrs));
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
        NodeEntry{jgraph.nodes[e.node_id].node, e.index, e.version});
  }
  return ret;
}

// save a graph to json
Graph SaveJSON(Graph src) {
  JSONGraph jgraph;
  jgraph.attrs = src.attrs;
  std::unordered_map<Node*, uint32_t> node2index;
  jgraph.node_row_ptr.push_back(0);
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
            JSONNode::Entry{node2index.at(e.node.get()), e.index, e.version});
      }
      for (const NodePtr& c : n->control_deps) {
        jnode.control_deps.push_back(node2index.at(c.get()));
      }
      jgraph.node_row_ptr.push_back(
          jgraph.node_row_ptr.back() + n->num_outputs());
      jgraph.nodes.emplace_back(std::move(jnode));
    });

  for (const NodeEntry& e : src.outputs) {
    jgraph.heads.push_back(
        JSONNode::Entry{node2index.at(e.node.get()), e.index, e.version});
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
DMLC_JSON_ENABLE_ANY(std::vector<std::string>, list_str);

}  // namespace
}  // namespace pass
}  // namespace nnvm
