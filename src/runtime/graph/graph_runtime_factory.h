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
 * \file tvm/runtime/graph_runtime_factory.h
 * \brief Graph runtime factory creating graph runtime.
 */

#ifndef TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_FACTORY_H_
#define TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_FACTORY_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "./graph_runtime.h"

namespace tvm {
namespace runtime {

class TVM_DLL GraphRuntimeFactory : public runtime::ModuleNode {
 public:
  /*!
   * \brief Construct the GraphRuntimeFactory.
   * \param graph_json The execution graph.
   * \param params The params of graph.
   * \param module_name The module name of graph.
   */
  GraphRuntimeFactory(const std::string& graph_json,
                      const std::unordered_map<std::string, tvm::runtime::NDArray>& params,
                      const std::string& module_name = "default");

  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const override { return "GraphRuntimeFactory"; }

  /*!
   * \brief Save the module to binary stream.
   * \param stream The binary stream to save to.
   */
  void SaveToBinary(dmlc::Stream* stream) override;

  /*!
   * \brief Create a specific runtime module
   * \param ctxs The context of the host and devices where graph nodes will be
   *  executed on.
   * \return created runtime module
   */
  Module RuntimeCreate(const std::vector<TVMContext>& ctxs);

  /*!
   * \brief Create a specific debug runtime module
   * \param ctxs The context of the host and devices where graph nodes will be
   *  executed on.
   * \return created debug runtime module
   */
  Module DebugRuntimeCreate(const std::vector<TVMContext>& ctxs);

  /*!
   * \brief Set params.
   * \param graph_runtime The graph runtime we want to set the params into.
   * \param params The graph params value we want to set.
   */
  void SetParams(GraphRuntime* graph_runtime,
                 const std::unordered_map<std::string, tvm::runtime::NDArray>& params) const {
    std::unordered_map<std::string, tvm::runtime::NDArray> value = params;
    // upload big arrays first to avoid memory issue in rpc mode
    std::vector<std::string> keys;
    for (const auto& p : value) {
      keys.emplace_back(p.first);
    }
    std::sort(std::begin(keys), std::end(keys),
              [&](const std::string& lhs, const std::string& rhs) -> bool {
                auto lhs_size = GetDataSize(value[lhs].ToDLPack()->dl_tensor);
                auto rhs_size = GetDataSize(value[rhs].ToDLPack()->dl_tensor);
                return lhs_size > rhs_size;
              });
    for (const auto& key : keys) {
      int in_idx = graph_runtime->GetInputIndex(key);
      if (in_idx >= 0) {
        graph_runtime->SetInput(in_idx, const_cast<DLTensor*>(value[key].operator->()));
      }
    }
  }

 protected:
  /*! \brief The execution graph. */
  std::string graph_json_;
  /*! \brief The params. */
  std::unordered_map<std::string, tvm::runtime::NDArray> params_;
  /*! \brief module name */
  std::string module_name_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_FACTORY_H_
