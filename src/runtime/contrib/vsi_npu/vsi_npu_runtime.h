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
#ifndef TVM_RUNTIME_CONTRIB_VSI_NPU_VSI_NPU_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_VSI_NPU_VSI_NPU_RUNTIME_H_

#include <tim/vx/context.h>
#include <tim/vx/graph.h>
#include <tim/vx/tensor.h>
#include <tvm/runtime/packed_func.h>

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace vsi_npu {
struct TensorSpecIR {
  tim::vx::QuantType quant_type;
  int32_t channel_dim;
  std::vector<float> scales;
  std::vector<int32_t> zps;

  tim::vx::DataType data_type;
  std::vector<uint32_t> shape;
  tim::vx::TensorAttribute attr;
};

class VsiNpuModule : public ModuleNode {
 public:
  VsiNpuModule(const std::shared_ptr<char>& nbg_buffer, uint32_t nbg_size,
               const std::vector<tim::vx::TensorSpec>& inputs_spec,
               const std::vector<tim::vx::TensorSpec>& outputs_spec)
      : compiled_nbg_(nbg_buffer),
        nbg_size_(nbg_size),
        inputs_(inputs_spec),
        outputs_(outputs_spec){};

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  const char* type_key() const override { return "vsi_npu"; }

  /*!
   * \brief Save a compiled network to a binary stream, which can then be
   * serialized to disk.
   * \param stream The stream to save the binary.
   * \note See LoadFromBinary for the serialization format.
   */
  void SaveToBinary(dmlc::Stream* stream) final;

  /*!
   * \brief Load a compiled network from stream.
   * \param strm The binary stream to load.
   * \return The created vsi-npu module.
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

 private:
  void SerializeTensorSpec(tim::vx::TensorSpec& t_spec, std::ostream& out);
  static tim::vx::TensorSpec DeSerializeTensorSpec(std::istream& in);
  // TODO: we need handle multiply nbg in production
  std::shared_ptr<char> compiled_nbg_;
  uint32_t nbg_size_;
  std::vector<tim::vx::TensorSpec> inputs_;
  std::vector<tim::vx::TensorSpec> outputs_;

  std::shared_ptr<tim::vx::Context> vx_context_;
  std::shared_ptr<tim::vx::Graph> vx_graph_;
};
}  // namespace vsi_npu
}  // namespace runtime
}  // namespace tvm

#endif