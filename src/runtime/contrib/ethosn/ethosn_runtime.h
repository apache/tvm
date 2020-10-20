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
 * \file ethosn_runtime.h
 * \brief Execution handling of Ethos-N command streams.
 */
#ifndef TVM_RUNTIME_CONTRIB_ETHOSN_ETHOSN_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_ETHOSN_ETHOSN_RUNTIME_H_

#include <tvm/runtime/packed_func.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ethosn_support_library/Support.hpp"

namespace tvm {
namespace runtime {
namespace ethosn {

namespace sl = ::ethosn::support_library;

struct OrderedCompiledNetwork {
  std::unique_ptr<sl::CompiledNetwork> cmm;
  std::string name;
  std::vector<uint32_t> inputs;
  std::vector<uint32_t> outputs;
};

class EthosnModule : public ModuleNode {
 public:
  /*!
   * \brief The Ethos-N runtime module.
   * \param cmms A vector of compiled networks with input/output orders.
   */
  explicit EthosnModule(std::vector<OrderedCompiledNetwork>* cmms);

  /*!
   * \brief Get a PackedFunc from the Ethos-N module.
   * \param name The name of the function.
   * \param sptr_to_self The ObjectPtr that points to this module node.
   * \return The function pointer when it is found, otherwise, PackedFunc(nullptr).
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;
  /*!
   * \brief Save a compiled network to a binary stream, which can then be
   * serialized to disk.
   * \param stream The stream to save the binary.
   * \note See EthosnModule::LoadFromBinary for the serialization format.
   */
  void SaveToBinary(dmlc::Stream* stream) final;
  /*!
   * \brief Load a compiled network from stream.
   * \param strm The binary stream to load.
   * \return The created Ethos-N module.
   * \note The serialization format is:
   *
   *       size_t : number of functions
   *       [
   *         std::string : name of function (symbol)
   *         std::string : serialized command stream
   *         size_t      : number of inputs
   *         std::vector : order of inputs
   *         size_t      : number of outputs
   *         std::vector : order of outputs
   *       ] * number of functions
   */
  static Module LoadFromBinary(void* strm);
  /*!
   * \brief Save a module to a specified path.
   * \param path Where to save the serialized module.
   */
  void SaveToFile(const std::string& path, const std::string& format) override;

  const char* type_key() const override { return "ethos-n"; }

 private:
  /*! \brief A map between ext_symbols (function names) and ordered compiled networks. */
  std::map<std::string, OrderedCompiledNetwork> network_map_;
};

}  // namespace ethosn
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_ETHOSN_ETHOSN_RUNTIME_H_
