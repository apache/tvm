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
 * \file src/runtime/json/json_runtime.h
 * \brief Utilities for json runtime.
 */

#ifndef TVM_RUNTIME_JSON_JSON_RUNTIME_H_
#define TVM_RUNTIME_JSON_JSON_RUNTIME_H_

#include <tvm/runtime/container.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>

#include "json_node.h"

namespace tvm {
namespace runtime {
namespace json {

/*!
 * \brief A json runtime that executes the serialized JSON format. This runtime
 * can be extended by user defined runtime for execution.
 */
class JSONRuntimeBase : public ModuleNode {
 public:
  explicit JSONRuntimeBase(const std::string& graph_json) {
    LoadGraph(graph_json);
  }

  // The type key of each subclass can be saved to the json file and them
  // used to create the specific runtime during deserialization.
  // virtual const char* type_key() const = 0;
  const char* type_key() const { return "json"; }

  virtual void Init() { LOG(FATAL) << "NYI"; }

  /*!
   * \brief Get a packed function.
   * \param name The name/symbol of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The packed function.
   */
  virtual PackedFunc GetFunction(const std::string& name,
                                 const ObjectPtr<Object>& sptr_to_self) {
    return PackedFunc();
  }

  // Run(TVMValue*,value, int* type_code, int nargs), or
  // Run(TVMArgs arg, TVMRetValue rv) ?
  virtual void Run() { LOG(FATAL) << "NYI"; }

  void SetInput(const std::string& name, const NDArray& data) {
    auto it = input_map_.find(name);
    CHECK(it != input_map_.end()) << "Not found input: " << name;
    SetInput(it->second, data);
  }

  void SetInput(uint32_t index, const NDArray& data) {
    CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
    uint32_t eid = EntryID(input_nodes_[index], 0);
    data_entry_[eid] = data;
  }

  size_t NumOutputs() const { return outputs_.size(); }

  ObjectRef GetOutput() {
    // Return the NDArray directly if there is only one outpput.
    if (NumOutputs() == 1) {
      uint32_t eid = EntryID(outputs_[0]);
      return data_entry_[eid];
    }

    // We need to return an ADTObj if there are multiple outputs.
    std::vector<ObjectRef> outs;
    for (size_t i = 0; i < NumOutputs(); i++) {
      uint32_t eid = EntryID(outputs_[i]);
      outs.push_back(data_entry_[eid]);
    }
    return ADT::Tuple(outs);
  }

 protected:
  void LoadGraph(const std::string& graph_json) {
    std::istringstream is(graph_json);
    dmlc::JSONReader reader(&is);
    this->Load(&reader);

    for (size_t i = 0; i < input_nodes_.size(); i++) {
      uint32_t nid = input_nodes_[i];
      std::string& name = nodes_[nid].name_;
      input_map_[name] = i;
    }
  }

  void Load(dmlc::JSONReader* reader) {
    reader->BeginObject();
    std::string key;
    while (reader->NextObjectItem(&key)) {
      if (key == "nodes") {
        reader->Read(&nodes_);
      } else if (key == "arg_nodes") {
        reader->Read(&input_nodes_);
      } else if (key == "node_row_ptr") {
        reader->Read(&node_row_ptr_);
      } else if (key == "heads") {
        reader->Read(&outputs_);
      } else {
        LOG(FATAL) << "Unknow key: " << key;
      }
    }
  }

  // Get the node entry index.
  uint32_t EntryID(uint32_t nid, uint32_t index) const {
    return node_row_ptr_[nid] + index;
  }

  // Get the node entry index.
  uint32_t EntryID(const JSONGraphNodeEntry& e) const {
    return EntryID(e.id_, e.index_);
  }

  // Number of node entries.
  uint32_t NumEntries() const {
    return node_row_ptr_.back();
  }

 protected:
  /*! \brief The json graph nodes. */
  std::vector<JSONGraphNode> nodes_;
  /*! \brief The input nodes, including variables and constants. */
  std::vector<uint32_t> input_nodes_;
  /*! \brief Used for quick entry indexing. */
  std::vector<uint32_t> node_row_ptr_;
  /*! \brief Output entries. */
  std::vector<JSONGraphNodeEntry> outputs_;
  /*! \brief Data of that entry. */
  std::vector<NDArray> data_entry_;
  /*! \brief Map the input name to index. */
  std::unordered_map<std::string, uint32_t> input_map_;
};

}  // namespace json
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_JSON_JSON_RUNTIME_H_
