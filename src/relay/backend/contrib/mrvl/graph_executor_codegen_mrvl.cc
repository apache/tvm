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
 * \file src/relay/backend/contrib/mrvl/graph_executor_codegen_mrvl.cc
 * \brief Marvell MLIP specific API
 */

#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../src/relay/backend/graph_executor_codegen.h"

namespace tvm {
namespace relay {
namespace backend {

using IntegerArray = Array<Integer>;
using ShapeVector = std::vector<std::vector<int64_t>>;
using GraphAttrs = std::unordered_map<std::string, dmlc::any>;
using GraphObjectPtr = std::shared_ptr<GraphNode>;

extern "C" ExternalJsonWriterCB* GetExternalJsonWriter();

/*! \brief Input Node */
class GraphInputNodeMrvlExt : public GraphInputNode {
 public:
  GraphInputNodeMrvlExt() : GraphInputNode() {}
  GraphInputNodeMrvlExt(const std::string& name, const GraphAttrs& attrs)
      : GraphInputNode(name, attrs) {}

  static std::shared_ptr<GraphNode> make_node_ptr(const std::string& name,
                                                  const GraphAttrs& attrs) {
    auto ptr = std::make_shared<GraphInputNodeMrvlExt>(name, attrs);
    return std::dynamic_pointer_cast<GraphNode>(ptr);
  }

  GraphNodeType Type() const override { return kGraphInputNodeExt; }

  void Save(dmlc::JSONWriter* writer) const override {
    const std::string op_name{"null"};
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_name);
    writer->WriteObjectKeyValue("name", this->name_);
    writer->WriteObjectKeyValue("inputs", std::list<int>());
    writer->WriteObjectKeyValue("attrs", this->attrs_);
    writer->EndObject();
  }
};

class GraphOpNodeMrvlExt : public GraphOpNode {
 public:
  GraphOpNodeMrvlExt() {}
  virtual ~GraphOpNodeMrvlExt() {}

  GraphNodeType Type() const override { return kGraphOpNodeExt; }

  void Load(dmlc::JSONReader* reader) override;
  void LoadAttrs(dmlc::JSONReader* reader);
  std::pair<std::string, GraphAttrs> GetLoadedGraphAttrs();
  std::string func_node_name_;
  GraphAttrs op_attrs_;
};

/*!
 * \brief Load a node in the json string.
 * \param reader The json reader.
 */
void GraphOpNodeMrvlExt::Load(dmlc::JSONReader* reader) {
  std::string tmp_name;
  std::vector<int64_t> tmp_int_arr;

  reader->BeginObject();
  std::string key;
  while (reader->NextObjectItem(&key)) {
    if (key == "op") {
      reader->Read(&tmp_name);
    } else if (key == "name") {
      reader->Read(&tmp_name);
    } else if (key == "inputs") {
      reader->BeginArray();
      ICHECK(reader->NextArrayItem()) << "invalid json format";
      reader->Read(&tmp_int_arr);
      if (reader->NextArrayItem()) {
        reader->Read(&tmp_int_arr);
        if (reader->NextArrayItem()) {
          reader->Read(&tmp_int_arr);
          ICHECK(!reader->NextArrayItem()) << "invalid json format";
        }
      }
    } else if (key == "attr" || key == "attrs") {
      this->LoadAttrs(reader);
    } else {
      LOG(FATAL) << "Unknown key: " << key;
    }
  }
}

/*!
 * \brief Load the attribute of a node in the json string.
 * \param reader The json reader.
 */
void GraphOpNodeMrvlExt::LoadAttrs(dmlc::JSONReader* reader) {
  std::string key;
  std::string tmp_str;
  GraphAttrs attrs;
  op_attrs_ = attrs;

  // - skip num_inputs and num_outputs (use originals)
  // - skip dtype for now
  // - skip shape
  reader->BeginObject();
  while (reader->NextObjectItem(&key)) {
    if (key == "num_inputs") {
      reader->Read(&tmp_str);
    } else if (key == "num_outputs") {
      reader->Read(&tmp_str);
    } else if (key == "dtype") {
      reader->BeginArray();
      std::vector<std::string> tmp_str_vec;
      ICHECK(reader->NextArrayItem());
      reader->Read(&tmp_str_vec);
      ICHECK(!reader->NextArrayItem());
    } else if (key == "shape") {
      reader->BeginArray();
      std::vector<std::vector<int64_t>> tmp_shape;
      ICHECK(reader->NextArrayItem());
      reader->Read(&tmp_shape);
      ICHECK(!reader->NextArrayItem());
    } else {
      reader->BeginArray();
      std::vector<std::string> tmp_str_vec;
      ICHECK(reader->NextArrayItem());
      reader->Read(&tmp_str_vec);
      op_attrs_[key] = tmp_str_vec;
      ICHECK(!reader->NextArrayItem());
      if (key == "func_node_name") {
        ICHECK(tmp_str_vec.size() == 1);
        func_node_name_ = tmp_str_vec[0];
      }
    }
  }
}

/*!
 * \brief return generated map<func_node_name, graph node attrs>
 */
std::pair<std::string, GraphAttrs> GraphOpNodeMrvlExt::GetLoadedGraphAttrs() {
  return std::pair<std::string, GraphAttrs>(func_node_name_, op_attrs_);
}

class MrvlExtJson {
 public:
  MrvlExtJson() {
    ICHECK(!GetExternalJsonWriter()->HasCallback()) << "ERROR: has registered callback";
    GetExternalJsonWriter()->RegisterCB(this, &MrvlExtJson::GetExternalJSON);
  }

  virtual ~MrvlExtJson() {}

  void GetExternalJSON(dmlc::JSONWriter* writer, Array<tvm::runtime::Module> external_mods,
                       std::vector<GraphObjectPtr> nodes, std::vector<GraphNodeRef> heads);

  void LoadExternalJsonAttrs(std::unordered_map<std::string, GraphAttrs>* external_attrs_map,
                             const Array<tvm::runtime::Module>& external_mods);
};

/*!
 * \brief Load External Json attrs map<func-name, graph-attrs>
 *
 * \param external_attrs_map: map to be generated
 * \param external_mods: array of external-codegen mods (one per external
 * composite func)
 */
void MrvlExtJson::LoadExternalJsonAttrs(
    std::unordered_map<std::string, GraphAttrs>* external_attrs_map,
    const Array<tvm::runtime::Module>& external_mods) {
  // retrieve attributes from each external composite graph
  for (size_t i = 0; i < external_mods.size(); ++i) {
    auto mod = external_mods[i];
    auto pfunc = mod.GetFunction("get_graph_json", false);
    std::string graph_json = pfunc();
    std::istringstream tmp_is(graph_json);
    dmlc::JSONReader tmp_reader(&tmp_is);

    std::vector<GraphOpNodeMrvlExt> tmp2_nodes;
    std::vector<int64_t> tmp_int_array;
    std::string key;
    tmp_reader.BeginObject();
    while (tmp_reader.NextObjectItem(&key)) {
      if (key == "nodes") {
        tmp_reader.Read(&tmp2_nodes);
      } else if (key == "arg_nodes") {
        tmp_reader.Read(&tmp_int_array);
      } else if (key == "node_row_ptr") {
        tmp_reader.Read(&tmp_int_array);
      } else if (key == "heads") {
        tmp_reader.BeginArray();
        ICHECK(tmp_reader.NextArrayItem()) << "invalid json format";
        tmp_reader.Read(&tmp_int_array);
        ICHECK(!tmp_reader.NextArrayItem()) << "invalid json format";
      } else {
        LOG(FATAL) << "Unknown key: " << key;
      }
    }
    std::pair<std::string, GraphAttrs> mrvl_node_attrs =
        tmp2_nodes[tmp2_nodes.size() - 1].GetLoadedGraphAttrs();
    external_attrs_map->insert({mrvl_node_attrs.first, mrvl_node_attrs.second});
  }
}

/*!
 * \brief Generate External Graph JSON
 *
 * \param writer json writer
 */
void MrvlExtJson::GetExternalJSON(dmlc::JSONWriter* writer,
                                  Array<tvm::runtime::Module> external_mods,
                                  std::vector<GraphObjectPtr> nodes,
                                  std::vector<GraphNodeRef> heads) {
  // retrieve attributes from each external composite graph
  std::unordered_map<std::string, GraphAttrs> external_attrs_map;
  LoadExternalJsonAttrs(&external_attrs_map, external_mods);

  /*! \brief nodes */
  std::vector<GraphObjectPtr> external_nodes = nodes;
  /*! \brief output of graph */
  std::vector<GraphNodeRef> external_heads = heads;

  for (size_t i = 0; i < external_nodes.size(); ++i) {
    auto node = external_nodes[i];
    if (node->Type() == kGraphOpNode) {
      // replace the op_attrs of this GraphOpNode node with its corresponding
      // external codegen node's attrs
      if (external_attrs_map.count(node->name_) == 1) {
        std::dynamic_pointer_cast<GraphOpNode>(node)->op_attrs_ = external_attrs_map[node->name_];
      }
    }
  }

  std::vector<size_t> arg_nodes;
  for (size_t i = 0; i < external_nodes.size(); ++i) {
    auto node = external_nodes[i];
    if (node->Type() == kGraphInputNode) {
      arg_nodes.push_back(i);
    }
  }
  size_t num_entry = 0;
  ShapeVector shapes;
  std::vector<size_t> storage_ids;
  std::vector<size_t> device_types;
  std::vector<std::string> dltypes;
  std::vector<size_t> node_row_ptr{0};
  for (size_t i = 0; i < external_nodes.size(); ++i) {
    auto node = external_nodes[i];
    const auto& shape_vec = dmlc::get<ShapeVector>(node->attrs_["shape"]);
    const auto& storage_id = dmlc::get<std::vector<int64_t>>(node->attrs_["storage_id"]);
    const auto& dtype_vec = dmlc::get<std::vector<std::string>>(node->attrs_["dtype"]);

    ICHECK_EQ(node->num_outputs_, shape_vec.size());
    num_entry += node->num_outputs_;

    shapes.insert(shapes.end(), shape_vec.begin(), shape_vec.end());
    dltypes.insert(dltypes.end(), dtype_vec.begin(), dtype_vec.end());
    storage_ids.insert(storage_ids.end(), storage_id.begin(), storage_id.end());
    if (node->attrs_.count("device_index")) {
      const auto& dev_types = dmlc::get<std::vector<int64_t>>(node->attrs_["device_index"]);
      device_types.insert(device_types.end(), dev_types.begin(), dev_types.end());
    }
    node_row_ptr.push_back(num_entry);

    if ((node->Type() == kGraphInputNode) && (external_attrs_map.size() > 0)) {
      ICHECK(dynamic_cast<GraphInputNode*>(node.get()));
      // copy GraphInputNode node to a GraphInputNodeMrvlExt node in order to
      //   use its own writer (i.e., by calling its Save() func)
      auto new_input_node_ptr = GraphInputNodeMrvlExt::make_node_ptr(node->name_, node->attrs_);
      external_nodes[i] = std::dynamic_pointer_cast<GraphNode>(new_input_node_ptr);

      // add "attrs": { "layer_name": [ "input" ] }
      std::vector<dmlc::any> layer_name_json_attr;
      layer_name_json_attr.emplace_back(std::string("input"));

      // add "attrs": { "data_layout": [ "NCHW" or "NHWC" or "NC" etc. ] }
      // TODO(ccjoechou): improve coverage to allow other networks
      bool is_NC = (!shape_vec.empty()) && (shape_vec[0].size() == 2);
      new_input_node_ptr->attrs_.clear();
      new_input_node_ptr->attrs_["layer_name"] = layer_name_json_attr;
      std::vector<dmlc::any> data_layout_json_attr;
      if (is_NC) {
        data_layout_json_attr.emplace_back(std::string("NC"));
      } else {
        data_layout_json_attr.emplace_back(std::string("NCHW"));
      }
      new_input_node_ptr->attrs_["data_layout"] = data_layout_json_attr;
    }
  }
  writer->BeginObject();
  writer->WriteObjectKeyValue("nodes", external_nodes);
  writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
  writer->WriteObjectKeyValue("heads", external_heads);
  std::unordered_map<std::string, std::vector<dmlc::any>> attrs;
  attrs["shape"].emplace_back(std::string("list_shape"));
  attrs["shape"].emplace_back(shapes);
  attrs["storage_id"].emplace_back(std::string("list_int"));
  attrs["storage_id"].emplace_back(storage_ids);
  if (device_types.size()) {
    attrs["device_index"].emplace_back(std::string("list_int"));
    attrs["device_index"].emplace_back(device_types);
  }
  attrs["dltype"].emplace_back(std::string("list_str"));
  attrs["dltype"].emplace_back(dltypes);
  writer->WriteObjectKeyValue("attrs", attrs);
  writer->WriteObjectKeyValue("node_row_ptr", node_row_ptr);
  writer->EndObject();
}

std::shared_ptr<MrvlExtJson> g_mrvl_ext_json;

extern "C" bool g_mrvlExtJsonObjInstantized;
bool g_mrvlExtJsonObjInstantized = false;

extern "C" void InstantiateMrvlExtJsonObj() {
  g_mrvl_ext_json = std::make_shared<MrvlExtJson>();
  g_mrvlExtJsonObjInstantized = true;
}

void MrvlClearFlag() { g_mrvlExtJsonObjInstantized = false; }
TVM_REGISTER_GLOBAL("relay.mrvl.clear_ext_json_flag").set_body_typed(MrvlClearFlag);

}  // namespace backend
}  // namespace relay
}  // namespace tvm

namespace dmlc {
namespace json {

// JSON utils to be template specialized for Mrvl BYOC GetExternalJSON() related extensions
template <typename T>
inline bool SameType(const dmlc::any& data) {
  return std::type_index(data.type()) == std::type_index(typeid(T));
}

template <>
struct Handler<std::shared_ptr<tvm::relay::backend::GraphNode>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::shared_ptr<tvm::relay::backend::GraphNode>& data) {
    data->Save(writer);
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::shared_ptr<tvm::relay::backend::GraphNode>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};

template <>
struct Handler<std::unordered_map<std::string, dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::unordered_map<std::string, dmlc::any>& data) {
    writer->BeginObject();
    for (const auto& kv : data) {
      auto k = kv.first;
      const dmlc::any& v = kv.second;
      if (SameType<std::string>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::string>(v));
      } else if (SameType<int>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<int>(v));
      } else if (SameType<std::vector<size_t>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<size_t>>(v));
      } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<std::vector<int64_t>>>(v));
      } else if (SameType<std::vector<std::string>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<std::string>>(v));
      } else if (SameType<std::vector<dmlc::any>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<dmlc::any>>(v));
      } else {
        LOG(FATAL) << "Value type not supported for key: " << k;
      }
    }
    writer->EndObject();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::unordered_map<std::string, dmlc::any>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};

template <>
struct Handler<std::vector<dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer, const std::vector<dmlc::any>& data) {
    writer->BeginArray();
    for (const auto& v : data) {
      if (SameType<std::string>(v)) {
        writer->WriteArrayItem(dmlc::get<std::string>(v));
      } else if (SameType<int>(v)) {
        writer->WriteArrayItem(dmlc::get<int>(v));
      } else if (SameType<std::vector<size_t>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<size_t>>(v));
      } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<std::vector<int64_t>>>(v));
      } else if (SameType<std::vector<std::string>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<std::string>>(v));
      } else {
        LOG(FATAL) << "Not supported";
      }
    }
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, std::vector<dmlc::any>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};

}  // namespace json
}  // namespace dmlc
