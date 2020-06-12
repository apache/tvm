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
 * \file src/runtime/json/json_runtime_driver.cc
 * \brief The driver for json runtime.
 */

#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <unordered_map>
#include <vector>
#include <string>

#include "json_node.h"
#include "json_runtime.h"

namespace tvm {
namespace runtime {
namespace json {

/*!
 * \brief The class represents a json runtime driver. It is mainly responsible
 * for 1) serializing and deserializing the json runtime artifacts, 2)
 * dispatching and invoking the actual runtime that intepretes the json
 * artifacts.
 */
class JSONRuntimeDriver : public ModuleNode {
 public:
  struct Subgraph {
    std::string symbol_name;
    std::string graph_json;
    std::unordered_map<std::string, NDArray> weights;
  };

  explicit JSONRuntimeDriver(const std::string& graph_json) {
    this->graph_json_ = graph_json;
    Deserialize();
  }

  const char* type_key() const { return "jsonruntime"; }

  /*!
   * \brief Get a packed function.
   * \param name The name/symbol of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The packed function.
   */
  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) {
    if (this->subgraphs_.count(name)) {
      return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
        auto json_rt = this->subgraphs_[name];
        auto* json_rt_node = static_cast<JSONRuntimeBase*>(json_rt.operator->());
        CHECK(json_rt_node);
        // Set input, how to make sure it is only invoked once? Likely we don't
        // really need this as we could directly set input when creating the
        // engine, but what if the input for each inference varies.
        // json_rt_node->SetInput();
        //
        // Execute the egine
        json_rt_node->Run();

        // Get the output, set rv or fill directly to args?
        *rv = json_rt_node->GetOutput();
      });
    } else {
      // Issue a warning when we don't find the symbol from the module. Note
      // we don't kill the execution here as the symbol may exist in other
      // runtime modules.
      LOG(WARNING) << "Cannot find " << name << " from json runtime";
      return PackedFunc();
    }
  }

  void Deserialize() {
    std::vector<Subgraph> subgraphs;
    dmlc::MemoryStringStream memstrm(&graph_json_);
    dmlc::Stream* strm = &memstrm;
    // Header
    uint64_t header;
    CHECK(strm->Read(&header)) << "Invalid serialized file format";

    // Compiler name
    std::string compiler_name;
    CHECK(strm->Read(&compiler_name)) << "Invalid serialized file format";

    uint64_t num_subgraphs;
    CHECK(strm->Read(&num_subgraphs)) << "Invalid serialized file format";
    // CHECK(header == kTVMJSONRuntimeMagic) << "Invalid serialized file format";

    for (uint64_t i = 0; i < num_subgraphs; i++) {
      Subgraph g;
      // Load the symbol for runtime lookup.
      std::string symbol_name;
      CHECK(strm->Read(&symbol_name)) << "Invalid serialized file format";
      g.symbol_name = symbol_name;

      // Load the graph representation.
      std::string json_graph;
      CHECK(strm->Read(&json_graph)) << "Invalid serialized file format";
      g.graph_json = json_graph;

      // Load the weights for the graph.
      uint64_t num_params;
      CHECK(strm->Read(&num_params)) << "Invalid serialized file format";

      std::vector<std::string> names;
      CHECK(strm->Read(&names)) << "Invalid serialized file format";
      CHECK_EQ(names.size(), num_params) << "Invalid serialized file format";

      for (size_t i = 0; i < static_cast<size_t>(num_params); i++) {
        NDArray tmp;
        tmp.Load(strm);
        g.weights[names[i]] = tmp;
      }
      subgraphs.push_back(g);
    }
    CreateSubgraphs(subgraphs, compiler_name);
  }

  // Create subgraphs for a specific runtime and cache it, therefore, we can
  // invoke them without the need to repeatedly create them at runtime.
  void CreateSubgraphs(const std::vector<Subgraph>& subgraphs,
                       const std::string& compiler_name) {
    // How do we know which runtime to create? Should we bake something in the
    // json to indicate this? i.e. we can register a runtime "runtime.ext.dnnl"
    // and save dnnl. Now we can just get it from the registry using dnnl. This
    // requires us to have single place to invoke different external codegens
    // and serialize them.
    //
    std::string ext_runtime_name = "runtime.ext." + compiler_name;
    auto pf = tvm::runtime::Registry::Get(ext_runtime_name);
    CHECK(pf) << "Failed to find the extern runtime for " << ext_runtime_name;
    for (const auto& sg : subgraphs) {
      CHECK_EQ(subgraphs_.count(sg.graph_json), 0U)
        << "Found duplicated symbol: " << sg.graph_json;

      Module ext_mod = (*pf)(sg.graph_json);
      const auto* json_rt_node = ext_mod.as<JSONRuntimeBase>();
      CHECK(json_rt_node);
      // Set up the params that are constants.
      for (const auto& it : sg.weights) {
        CallPakcedFunc(ext_mod, "set_input", it.first, it.second);
      }
      // Init the engine
      CallPakcedFunc(ext_mod, "init");

      subgraphs_[sg.graph_json] = ext_mod;
    }
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string graph;
    stream->Read(&graph);
    auto n = make_object<JSONRuntimeDriver>(graph);
    return Module(n);
  }

  void SaveToBinary(dmlc::Stream* stream) override {
    stream->Write(this->graph_json_);
  }

 private:
  template <typename... Args>
  void CallPakcedFunc(Module mod, const std::string& name, Args... args) {
    auto pf = mod.GetFunction(name);
    pf(std::forward<Args>(args)...);
  }

  /*! \brief The graph json. Weights are also baked in. */
  std::string graph_json_;
  /*!
   * \brief Cache the created runtime module that can be directly invoked.
   *
   * The runtime could be a csource runtime or a any user defined runtime that
   * is extend from the JSONRuntimeBase class.
   */
  std::unordered_map<std::string, Module> subgraphs_;
};

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_jsonruntime")
.set_body_typed(JSONRuntimeDriver::LoadFromBinary);

runtime::Module JSONRuntimeDriverCreate(std::string graph_json) {
  auto n = make_object<JSONRuntimeDriver>(graph_json);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.JSONRuntimeDriverCreate")
.set_body_typed(JSONRuntimeDriverCreate);

}  // namespace json
}  // namespace runtime
}  // namespace tvm

