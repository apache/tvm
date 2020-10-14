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
* \brief Contains TensorRTBuilder class which can be used to convert a relay
* program into a TRT engine which can be used for inference.
*/

#ifndef TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_BUILDER_H_
#define TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_BUILDER_H_

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>

#include <string>
#include <unordered_map>
#include <vector>
#include "NvInfer.h"

#define TRT_VERSION_GE(major, minor, patch)                    \
  ((NV_TENSORRT_MAJOR > major) ||                              \
  (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR > minor) || \
  (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR == minor && \
  NV_TENSORRT_PATCH >= patch))

#include "tensorrt_logger.h"

namespace tvm {
namespace runtime {

/*!
 * \brief The product of TensorRTBuilder which provides everything needed to
 * perform inference.
 */
struct TrtEngineAndContext {
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;
  std::vector<std::string> inputs;
  std::vector<bool> input_is_baked;
  std::vector<std::string> outputs;
  std::vector<runtime::NDArray> device_mem_buffers;
};

}  // namespace runtime

namespace relay {
namespace contrib {

/*!
 * \brief An input to a op may be either kTensor in the case of nvifner::ITensor
 * or kWeight for nvinfer1::Weights.
 */
enum TrtInputType {
  kTensor,
  kWeight,
};

/*!
 * \brief An input to a TrtOpConverter. The type of the input is either kTensor
 * or kWeight. For kTensor, "tensor" contains the input tensor. For kWeight,
 * "weight" contains the input weight and "weight_shape" contains the shape.
 */
struct TrtOpInput {
  TrtInputType type;
  nvinfer1::ITensor* tensor;
  nvinfer1::Weights weight;
  std::vector<int> weight_shape;

  explicit TrtOpInput(nvinfer1::ITensor* tensor)
      : tensor(tensor), type(kTensor) {}
  TrtOpInput(nvinfer1::Weights weight, const std::vector<int>& shape)
      : weight(weight), type(kWeight), weight_shape(shape) {}
};

/*!
 * \brief An ExprVisitor to convert a relay expression into a TensorRT engine
 * and execution context.
 */
class TensorRTBuilder : public ExprVisitor {
 public:
  /*!
   * \brief Create TensorRT builder.
   * \param args Inputs to this execution.
   */
  explicit TensorRTBuilder(runtime::TensorRTLogger* logger, const std::vector<DLTensor*>& args,
                           size_t max_workspace_size, bool use_implicit_batch_);

  void VisitExpr_(const VarNode* node) final;

  void VisitExpr_(const ConstantNode* node) final;

  void VisitExpr_(const TupleGetItemNode* op) final;

  void VisitExpr_(const TupleNode* op) final;

  void VisitExpr_(const CallNode* call) final;

  /*!
   * \brief Convert Expr into TensorRT.
   * \param expr The relay expression.
   * \return TRT engine, context, and input/output information.
   */
  runtime::TrtEngineAndContext BuildEngine(const Function& func);

 private:
  /*!
   * \brief Helper function fto convert NDArray to TRT Weights.
   * \param array NDArray containing data.
   * \param src_device Which device the data is expected to be on.
   * \return Newly created weights
   */
  nvinfer1::Weights GetNdArrayAsWeights(const runtime::NDArray& array,
                                        DLDeviceType src_device);

  /*!
   * \brief Helper function fto convert DLTensor to TRT Weights.
   * \param dptr Pointer to DLTensor containing data.
   * \param src_device Which device the data is expected to be on.
   * \return Newly created weights
   */
  nvinfer1::Weights GetDLTensorAsWeights(DLTensor* dptr,
                                         DLDeviceType src_device);

  nvinfer1::ITensor* AddInput(const std::string& tensor_name, const Type& type);

  /*! \brief Gets value from execution args and converts to constant weight
   * stored in node_output_map_ with node as the key. */
  void GetInputAsWeights(const VarNode* node);

  /*! \brief Gets value from ConstantNode data and converts to constant weight
   * stored in node_output_map_ with node as the key. */
  void GetConstantAsWeights(const ConstantNode* node);

  /*! \brief Temporary workaround for transposed weights. */
  void GetInputAsTransposedWeights(const CallNode* transpose,
                                   const VarNode* node);

  /*! \brief Deallocates weights and destroys network definition. */
  void CleanUp();

  /*! \brief Initializes network_input_names_, network_input_map_ and
   * network_input_is_baked_ based on function parameters. */
  void ProcessInputs(const Function& expr);

  /*! \brief Populates network_output_names_ from the final outputs of the
   * processed expr. */
  void ProcessOutputs(const Expr& expr);

  /*! \brief Maps a node to its outputs. */
  std::unordered_map<const ExprNode*, std::vector<TrtOpInput>> node_output_map_;

  /*! \brief TensorRT builder. */
  nvinfer1::IBuilder* builder_;

#if TRT_VERSION_GE(6, 0, 1)
  /*! \brief TensorRT builder config. */
  nvinfer1::IBuilderConfig* config_;
#endif

  /*! \brief TensorRT network definition. */
  nvinfer1::INetworkDefinition* network_;

  /*! \brief List of all weights held in memory. */
  std::vector<nvinfer1::Weights> trt_weights_;

  /*! \brief Execution inputs from this invocation. */
  const std::vector<DLTensor*>& execution_args_;

  /*! \brief Batch size of inputs from this invocation. */
  int batch_size_;

  /*! \brief Max workspace size in bytes for TRT. */
  size_t max_workspace_size_;

  /*! \brief Whether to use implicit batch mode. */
  bool use_implicit_batch_;

  /*! \brief Input names in same order as execution args during runtime. Some of
   * these are not actual input bindings in the TRT engine - use
   * network_input_is_baked_ to find out which. */
  std::vector<std::string> network_input_names_;

  /*! \brief Maps input name to execution args index. */
  std::unordered_map<std::string, int> network_input_map_;

  /*! \brief True if the corresponding input is baked into the TensorRT engine
   * and therefore should not be included in the input bindings during
   * execution. */
  std::vector<bool> network_input_is_baked_;

  /*! \brief Output names in same order as execution args during runtime. */
  std::vector<std::string> network_output_names_;
};

/*!
 * \brief Helper function for GetInputAsTransposedWeights to transpose 4-D
 * weights.
 * \param original_shape Shape of weight before transpose.
 * \param output_strides Multipliers for each index to compute output index in
 * flat buffer. Must be of length 4.
 * \param input_values The original weight values.
 * \param output_values Buffer where transposed values will be placed.
 */
void TransposeWeights4D(const std::vector<int>& original_shape,
                        const int* output_strides, const float* input_values,
                        float* output_values);

/*!
 * \brief Helper function for GetInputAsTransposedWeights to transpose CK to KC.
 * \param original_shape Shape of weight before transpose.
 * \param input_values The original weight values.
 * \param output_values Buffer where transposed values will be placed.
 */
void TransposeWeights2D(const std::vector<int>& original_shape,
                        const float* input_values, float* output_values);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_BUILDER_H_
