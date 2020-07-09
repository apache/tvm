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

#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <utility>

#include "../../../relay/backend/contrib/acl/acl_api.h"
#include "../../file_util.h"

#ifdef TVM_GRAPH_RUNTIME_ACL
#include <arm_compute/runtime/MemoryManagerOnDemand.h>
#include <arm_compute/runtime/OffsetLifetimeManager.h>
#include <arm_compute/runtime/PoolManager.h>

#include "acl_allocator.h"
#include "acl_kernel.h"
#endif

namespace tvm {
namespace runtime {

namespace api = relay::contrib::acl;

class ACLModule : public ModuleNode {
 public:
  /*!
   * \brief The ACL runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param serialized_graphs A vector of (external symbol, serialized JSON subgraph) pairs.
   */
  explicit ACLModule(const std::vector<std::pair<std::string, std::string>>& serialized_functions) {
#ifdef TVM_GRAPH_RUNTIME_ACL
    auto lifetime_mgr = std::make_shared<arm_compute::OffsetLifetimeManager>();
    auto pool_mgr = std::make_shared<arm_compute::PoolManager>();
    auto mm = std::make_shared<arm_compute::MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);
    int num_pools = 0;
#endif

    for (const auto& it : serialized_functions) {
      std::string serialized_function = it.second;
      auto ds = api::DeserializeSubgraph(&serialized_function);
      this->deserialized_functions_.emplace_back(it.first, ds);

#ifdef TVM_GRAPH_RUNTIME_ACL
      this->subgraph_cache_[it.first] =
          std::make_shared<contrib::acl::CachedLayer>(ds.first, ds.second, &this->allocator_, mm);
      if (this->subgraph_cache_[it.first]->IsMemoryManaged()) num_pools++;
#endif
    }
#ifdef TVM_GRAPH_RUNTIME_ACL
    // Allocate working memory for layers.
    if (num_pools > 0) mm->populate(this->allocator_, num_pools);
#endif
  }

  /*!
   * \brief Get a PackedFunc from the ACL module.
   *
   * \param name The name of the function.
   * \param sptr_to_self The ObjectPtr that points to this module node.
   * \return The function pointer when it is found, otherwise, PackedFunc(nullptr).
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
#ifdef TVM_GRAPH_RUNTIME_ACL
    if (this->subgraph_cache_.find(name) != this->subgraph_cache_.end()) {
      return PackedFunc([sptr_to_self, this, name](TVMArgs args, TVMRetValue* rv) {
        *rv = tvm::runtime::ACLModule::Inference(args, this->subgraph_cache_[name].get());
      });
    }
#endif
    return PackedFunc(nullptr);
  }

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* type_key() const override { return "acl"; }

  /*!
   * \brief Unpack inputs and outputs and run inference on a given layer.
   *
   * \param args Access inputs and outputs.
   * \param function The layer to execute inference on.
   * \return Status of inference.
   */
#ifdef TVM_GRAPH_RUNTIME_ACL
  static bool Inference(tvm::runtime::TVMArgs args, contrib::acl::CachedLayer* function) {
    // Unpack parameters
    int argc = 0;
    std::vector<DLTensor*> inputs;
    for (size_t i = 0; i < function->GetNumInputs(); i++) {
      inputs.push_back(args[argc++]);
    }
    std::vector<DLTensor*> outputs;
    for (; argc < args.size(); argc++) {
      outputs.push_back(args[argc]);
    }
    return function->Inference(inputs, outputs);
  }
#endif

  /*!
   * \brief Save a compiled network to a binary stream, which can then be
   * serialized to disk.
   *
   * \param stream The stream to save the binary.
   */
  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(this->deserialized_functions_.size());
    for (const auto& it : this->deserialized_functions_) {
      stream->Write(it.first);
      std::pair<api::JSONSubGraph, std::vector<NDArray>> subgraph_pair = it.second;
      std::string serialized_function =
          api::SerializeSubgraph(subgraph_pair.first, subgraph_pair.second);
      stream->Write(serialized_function);
    }
  }

  /*!
   * \brief Load a compiled network from stream.
   *
   * \param strm The binary stream to load.
   * \return The created ACL module.
   */
  static Module LoadFromBinary(void* strm) {
    auto stream = static_cast<dmlc::Stream*>(strm);
    size_t func_count;
    stream->Read(&func_count);
    std::vector<std::pair<std::string, std::string>> serialized_functions;
    for (unsigned int i = 0; i < func_count; i++) {
      std::string ext_symbol;
      std::string serialized_function;
      stream->Read(&ext_symbol);
      stream->Read(&serialized_function);
      serialized_functions.emplace_back(std::make_pair(ext_symbol, serialized_function));
    }
    auto n = make_object<ACLModule>(serialized_functions);
    return Module(n);
  }

  /*!
   * \brief Save a module to a specified path.
   *
   * \param path Where to save the serialized module.
   * \param format The format of the file.
   */
  void SaveToFile(const std::string& path, const std::string& format) override {
    std::string data;
    dmlc::MemoryStringStream writer(&data);
    dmlc::SeekStream* strm = &writer;
    SaveToBinary(strm);
    SaveBinaryToFile(path, data);
  }

  /*!
   * \brief Create a module from a file.
   *
   * \param path The path of the file containing the serialized module.
   * \return The created ACL module.
   */
  static Module LoadFromFile(const std::string& path) {
    std::string data;
    LoadBinaryFromFile(path, &data);
    dmlc::MemoryStringStream reader(&data);
    return LoadFromBinary(&reader);
  }

  /*!
   * \brief Get the JSON generated by codegen.
   *
   * \param format the format to return (only JSON for the time being)
   * \return A string of JSON.
   */
  std::string GetSource(const std::string& format) override {
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    for (const auto& it : deserialized_functions_) {
      writer.WriteObjectKeyValue(it.first, it.second.first);
    }
    writer.EndObject();
    return os.str();
  }

 private:
  /* \brief A vector of (external symbol, serialized JSON subgraph) pairs. */
  std::vector<std::pair<std::string, std::pair<api::JSONSubGraph, std::vector<NDArray>>>>
      deserialized_functions_;

#ifdef TVM_GRAPH_RUNTIME_ACL
  /* \brief A map between ext_symbols (function names) and an ACL subgraph.
   * \note Currently only a single op per subgraph is supported. Hence mapping to
   * cached layer.*/
  std::map<std::string, std::shared_ptr<contrib::acl::CachedLayer>> subgraph_cache_;
  /*! \brief Allow ACL functions to request auxiliary memory from TVM. */
  contrib::acl::ACLAllocator allocator_;
#endif
};

TVM_REGISTER_GLOBAL("runtime.module.loadfile_acl").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = ACLModule::LoadFromFile(args[0]);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_acl").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = ACLModule::LoadFromBinary(args[0]);
});

}  // namespace runtime
}  // namespace tvm
