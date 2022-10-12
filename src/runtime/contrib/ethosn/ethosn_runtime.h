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
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "ethosn_driver_library/Network.hpp"
#include "ethosn_support_library/Support.hpp"

namespace tvm {
namespace runtime {
namespace ethosn {

namespace sl = ::ethosn::support_library;
namespace dl = ::ethosn::driver_library;

struct OrderedCompiledNetwork {
  std::unique_ptr<sl::CompiledNetwork> compiled_cmm;
  std::unique_ptr<dl::Network> runtime_cmm;
  std::string name;
  std::vector<uint32_t> inputs;
  std::vector<uint32_t> outputs;
  std::vector<uint32_t> input_sizes;
  std::vector<uint32_t> output_sizes;
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
   *         std::vector : buffer sizes for inputs
   *         size_t      : number of outputs
   *         std::vector : order of outputs
   *         std::vector : buffer sizes for outputs
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

/*!
 * \brief Error codes for evaluating the result of inference on the NPU.
 */
enum class InferenceWaitErrorCode { kSuccess = 0, kTimeout = 1, kError = 2 };

/*!
 * \brief A helper class holding the status of inference on the NPU and
 * associated error message(s) if any occurred.
 *
 * Similar to the implementation of 'WaitStatus' in the driver stack:
 * https://github.com/ARM-software/ethos-n-driver-stack/blob/22.08/armnn-ethos-n-backend/workloads/EthosNPreCompiledWorkload.cpp#L48
 */
class InferenceWaitStatus {
 public:
  InferenceWaitStatus() : error_code_(InferenceWaitErrorCode::kSuccess), error_description_("") {}

  explicit InferenceWaitStatus(InferenceWaitErrorCode errorCode, std::string errorDescription = "")
      : error_code_(errorCode), error_description_(errorDescription) {}

  InferenceWaitStatus(const InferenceWaitStatus&) = default;
  InferenceWaitStatus(InferenceWaitStatus&&) = default;
  InferenceWaitStatus& operator=(const InferenceWaitStatus&) = default;
  InferenceWaitStatus& operator=(InferenceWaitStatus&&) = default;

  explicit operator bool() const { return error_code_ == InferenceWaitErrorCode::kSuccess; }
  InferenceWaitErrorCode GetErrorCode() const { return error_code_; }
  std::string GetErrorDescription() const { return error_description_; }

 private:
  InferenceWaitErrorCode error_code_;
  std::string error_description_;
};

}  // namespace ethosn
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_ETHOSN_ETHOSN_RUNTIME_H_
