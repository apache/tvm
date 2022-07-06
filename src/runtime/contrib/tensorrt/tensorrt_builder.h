/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/tensorrt/tensorrt_builder.h
 * \brief The TensorRTBuilder class can be used to convert a JSONRuntime graph into a TRT engine
 * which can be used for inference.
 */

#ifndef TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_BUILDER_H_
#define TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_BUILDER_H_

#include <tvm/runtime/ndarray.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_node.h"
#include "NvInfer.h"
#include "tensorrt_logger.h"
#include "tensorrt_ops.h"

namespace tvm {
namespace runtime {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

/*!
 * \brief The product of TensorRTBuilder which provides everything needed to
 * perform inference.
 */
struct TensorRTEngineAndContext {
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

/*!
 * \brief Converts a JSONRuntime graph into a TensorRT engine and execution context. Inputs,
 * constants, layers, and outputs can be added to construct the TensorRT network definition.
 * BuildEngine() will then use the network definition to build the TensorRT engine and context which
 * can be used to run inference - this phase can take a long time because TensorRT will query the
 * performance of all available kernels and fusions to optimize the engine.
 */
class TensorRTBuilder {
 public:
  /*!
   * \brief Create TensorRT builder.
   * \param logger TensorRT logger to use for errors and warnings.
   * \param max_workspace_size Workspace size parameter for TensorRT engine build phase.
   * \param use_implicit_batch Whether to use implicit batch mode (default)
   * \param use_fp16 Whether to automatically convert a model to fp16
   * \param batch_size If use_implicit_batch,
   */
  TensorRTBuilder(TensorRTLogger* logger, const std::vector<const DLTensor*>& data_entry,
                  size_t max_workspace_size, bool use_implicit_batch, bool use_fp16, int batch_size,
                  nvinfer1::IInt8Calibrator* calibrator = nullptr);

  /*!
   * \brief Add TensorRT input(s) for input node in network definition.
   * \param nid The input node id.
   * \param entry_id The index into data_entry_ for first entry in node.
   * \param node The input node.
   */
  void AddInput(int nid, uint32_t entry_id, const JSONGraphNode& node);

  /*!
   * \brief Add TensorRT weight for input constant in network definition.
   * \param nid The input node id.
   * \param node The data tensor on CPU.
   */
  void AddConstant(int nid, const DLTensor* data);

  /*!
   * \brief Add TensorRT layer for op node in network definition.
   * \param nid The input node id.
   * \param node The op node.
   */
  void AddLayer(int nid, const JSONGraphNode& node);

  /*!
   * \brief Mark TensorRT output in network definition.
   * \param entry The output node entry.
   * \param entry_id The output node entry id.
   */
  void AddOutput(const JSONGraphNodeEntry& entry, uint32_t entry_id);

  /*!
   * \brief Takes network definition and "compiles" a TensorRT engine which can be used for
   * inference. This step is time confusing.
   * \return TRT engine, context, and input/output information.
   */
  TensorRTEngineAndContext BuildEngine();

 private:
  /*! \brief Convert a DLTensor to a TensorRT weight. */
  nvinfer1::Weights GetDLTensorAsWeights(const DLTensor* dptr, DLDeviceType src_device);

  /*! \brief Convert an input to a Tensor if it is a Weight */
  nvinfer1::ITensor* GetInputAsTensor(const TensorRTOpInput& input);

  /*! \brief Clean up resources used to create engine. */
  void CleanUp();

  /*! \brief Maps a node to its outputs. */
  std::unordered_map<int, std::vector<TensorRTOpInput>> node_output_map_;

  /*! \brief TensorRT builder. */
  nvinfer1::IBuilder* builder_ = nullptr;

#if TRT_VERSION_GE(6, 0, 1)
  /*! \brief TensorRT builder config. */
  nvinfer1::IBuilderConfig* config_ = nullptr;
#endif

  /*! \brief TensorRT network definition. */
  nvinfer1::INetworkDefinition* network_ = nullptr;

  /*! \brief List of all weights held in memory. */
  std::vector<nvinfer1::Weights> trt_weights_;

  /*! \brief Input and output tensors from TVM. */
  const std::vector<const DLTensor*>& data_entry_;

  /*! \brief Map TensorRT binding name to index in data_entry_. */
  std::unordered_map<std::string, uint32_t> entry_id_map_;

  /*! \brief Max workspace size in bytes for TRT. */
  size_t max_workspace_size_;

  /*! \brief Whether to use implicit batch mode. */
  bool use_implicit_batch_;

  /*! \brief Whether to automatically convert model to 16-bit floating point precision. */
  bool use_fp16_;

  /*! \brief whether to automatically convert model to int8 precision */
  bool use_int8_;

  /*! \brief Batch size to optimize for. */
  int batch_size_;

  /*! \brief Input names. */
  std::vector<std::string> network_input_names_;

  /*! \brief Output names. */
  std::vector<std::string> network_output_names_;

  /*! \brief calibrator pointer to add batch data when using int8 mode */
  /*! \brief pointer will be nullptr when it is fp16 or fp32 precision */
  nvinfer1::IInt8Calibrator* calibrator_;
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_BUILDER_H_
