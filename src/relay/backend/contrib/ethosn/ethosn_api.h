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

#ifndef TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_API_H_
#define TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_API_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ethosn_support_library/Support.hpp"
#include "ethosn_support_library/SupportQueries.hpp"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosn {

namespace sl = ::ethosn::support_library;

struct ConvolutionParams {
  sl::ConvolutionInfo conv_info;
  sl::TensorInfo activation_info;
  sl::TensorInfo weights_info;
  sl::TensorInfo bias_info;
  void* raw_weights = nullptr;
  void* raw_bias = nullptr;
  bool is_depthwise = false;
};

struct FullyConnectedParams {
  sl::FullyConnectedInfo fc_info;
  sl::TensorInfo input_info;
  sl::TensorInfo weights_info;
  sl::TensorInfo bias_info;
  void* raw_weights = nullptr;
  void* raw_bias = nullptr;
};

struct MaxPool2DParams {
  sl::PoolingInfo pool_info = sl::PoolingInfo(0, 0, 0, 0, sl::Padding(), sl::PoolingType::MAX);
  sl::TensorInfo input_info;
};

struct AvgPool2DParams {
  sl::PoolingInfo pool_info = sl::PoolingInfo(0, 0, 0, 0, sl::Padding(), sl::PoolingType::AVG);
  sl::TensorInfo input_info;
};

struct ReshapeParams {
  sl::TensorShape new_shape{};
  sl::TensorInfo input_info;
};

struct AdditionParams {
  sl::QuantizationInfo output_quantization_info;
  sl::TensorInfo lhs_info;
  sl::TensorInfo rhs_info;
};

struct SigmoidParams {
  sl::TensorInfo input_info;
};

struct ConcatenateParams {
  sl::QuantizationInfo qInfo;
  sl::ConcatenationInfo concat_info = sl::ConcatenationInfo(1, qInfo);
  std::vector<sl::TensorInfo> input_infos;
};

struct SplitParams {
  sl::SplitInfo split_info = sl::SplitInfo(0, {});
  sl::TensorInfo input_info;
};

struct DepthToSpaceParams {
  sl::DepthToSpaceInfo depth_info = sl::DepthToSpaceInfo(0);
  sl::TensorInfo input_info;
};

struct ReluParams {
  sl::ReluInfo relu_info;
  sl::TensorInfo input_info;
};

/*!
 * \brief A wrapper around std::stringstream to build an EthosnError.
 */
class ErrStrm {
 public:
  template <typename T>
  ErrStrm& operator<<(const T& val) {  // NOLINT(*)
    stream_ << val;
    return *this;
  }

 private:
  std::stringstream stream_;
  friend class EthosnError;
};

/*!
 * \brief Custom error class for storing error messages produced
 * during compilation for Ethos-N.
 */
class EthosnError {
 public:
  /*! \brief Default constructor */
  EthosnError() {}
  /*!
   * \brief Construct error from an Array of Strings
   * \param msgs The messages
   */
  explicit EthosnError(const Array<String>& msgs) : msgs(msgs) {}
  /*!
   * \brief Construct error from a String
   * \param msg The message
   */
  explicit EthosnError(const String& msg) { msgs.push_back(msg); }
  /*!
   * \brief Construct error from an ErrStrm
   * \param err The ErrStrm
   */
  explicit EthosnError(const ErrStrm& err) : EthosnError(err.stream_.str()) {}

  /*! \return Whether there are any error messages */
  explicit operator bool() const { return !msgs.empty(); }

  /*! \brief Add together two errors to give a single error with all the msgs */
  EthosnError& operator+=(const EthosnError& other) {
    msgs.insert(msgs.end(), other.msgs.begin(), other.msgs.end());
    return *this;
  }

  /*! \brief The error messages */
  Array<String> msgs;
};

/*!
 * \brief Functions to interact with Support Library's API including the
 * translation of Relay ops/composite functions into Support Library
 * equivalents.
 */
class EthosnAPI {
 public:
  /*! \brief Extract the Support Library convolution params from an ethos-n.qnn_conv2d func */
  static EthosnError QnnConv2d(const Expr& expr, ConvolutionParams* params);
  /*! \brief Extract the Support Library dense params from an ethos-n.qnn_fc func */
  static EthosnError QnnFullyConnected(const Expr& expr, FullyConnectedParams* params);
  /*! \brief Extract the Support Library max_pool2d params from a Relay max_pool2d call */
  static EthosnError MaxPool2D(const Expr& expr, MaxPool2DParams* params);
  /*! \brief Extract the Support Library avg_pool params from a Relay ethos-n.qnn_avg_pool2d func */
  static EthosnError AvgPool2D(const Expr& expr, AvgPool2DParams* params);
  /*! \brief Extract the Support Library reshape params from a Relay reshape call */
  static EthosnError Reshape(const Expr& expr, ReshapeParams* params);
  /*! \brief Extract the Support Library addition params from a Relay qnn.addition call */
  static EthosnError Addition(const Expr& expr, AdditionParams* params);
  /*! \brief Extract the Support Library sigmoid params from a Relay an ethos-n.qnn_sigmoid func */
  static EthosnError Sigmoid(const Expr& expr, SigmoidParams* params);
  /*! \brief Extract the Support Library concatenate params from a Relay qnn.concatenate call */
  static EthosnError Concatenate(const Expr& expr, ConcatenateParams* params);
  /*! \brief Extract the Support Library split params from a Relay split call */
  static EthosnError Split(const Expr& expr, SplitParams* params);
  /*! \brief Extract the Support Library depth_to_space params from a Relay depth_to_space call */
  static EthosnError DepthToSpace(const Expr& expr, DepthToSpaceParams* params);
  /*! \brief Extract the Support Library relu params from a Relay relu call */
  static EthosnError Relu(const Expr& expr, ReluParams* params);

 private:
  /*! \brief Convert a TVM IndexExpr array to a SL tensor shape */
  static EthosnError Tvm2Npu(const Array<IndexExpr>& shape, sl::TensorShape* npu_shape);
  /*! \brief Convert a TVM data type to a SL data type */
  static EthosnError Tvm2Npu(const tvm::DataType& dtype, sl::DataType* data_type);
  /*! \brief Convert TVM 1D padding to SL padding */
  static EthosnError Tvm2Npu(const Array<IndexExpr>& padding, sl::Padding* npu_padding);
  /*! \brief Convert TVM 1D striding to SL striding */
  static EthosnError Tvm2Npu(const Array<IndexExpr>& strides, sl::Stride* npu_stride);
  /*! \brief Convert TVM data format to SL data format */
  static EthosnError Tvm2Npu(const std::string& dformat, sl::DataFormat* data_format);
  /*! \brief Convert TVM size array for pooling size to x and y values */
  static EthosnError Tvm2Npu(const Array<IndexExpr>& size, uint32_t* x, uint32_t* y);
  /*! \brief Convert TVM quantization info to SL quantization info */
  static EthosnError Tvm2Npu(int32_t zero_point, float scale, sl::QuantizationInfo* npu_qinfo);
  /*! \brief Convert TVM 2D padding to SL padding */
  static EthosnError Tvm2Npu(const Array<Array<Integer>>& padding, sl::Padding* npu_padding);
  /*! \brief Convert a TVM Integer array to a SL tensor shape */
  static EthosnError Tvm2Npu(const Array<Integer>& shape, sl::TensorShape* npu_shape);
  /*! \brief Convert a TVM pooling call to SL pooling information */
  static EthosnError Pool2d(const Call& pool, Array<IndexExpr> size, Array<IndexExpr> strides,
                            Array<IndexExpr> padding, sl::PoolingType pooling_type,
                            sl::PoolingInfo* pool_info, sl::TensorInfo* input_info,
                            std::string layout);

  // Convert an array of IntImmNodes into ValueT
  // IndexT type of Array indexing variable
  // ValueT type of resulting value
  template <typename IndexT, typename ValueT>
  static EthosnError AsArray(const Array<IndexT>& arr, std::array<ValueT, 4>* v);

  // Get a T from a constant represented by a NDArray.
  template <typename T>
  static EthosnError AsConstant(const Expr& expr, T* out);
};

}  // namespace ethosn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_ETHOSN_ETHOSN_API_H_
