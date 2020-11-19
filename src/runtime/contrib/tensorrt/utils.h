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
 * \file runtime/contrib/tensorrt/utils.h
 * \brief Helper functions used by TensorRTBuilder or TrtOpConverters.
 */

#ifndef TVM_RUNTIME_CONTRIB_TENSORRT_UTILS_H_
#define TVM_RUNTIME_CONTRIB_TENSORRT_UTILS_H_

#include "NvInfer.h"

#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <string>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief Helper function to convert an vector to TRT Dims.
 * \param vec Vector.
 * \return TRT Dims.
 */
template <typename T>
nvinfer1::Dims VectorToTrtDims(const std::vector<T>& vec) {
  nvinfer1::Dims dims;
  // Dims(nbDims=0, d[0]=1) is used to represent a scalar in TRT.
  dims.d[0] = 1;
  dims.nbDims = vec.size();
  for (size_t i = 0; i < vec.size(); ++i) {
    dims.d[i] = vec[i];
  }
  return dims;
}

/*!
 * \brief Helper function to convert TRT Dims to vector.
 * \param vec TRT Dims.
 * \return Vector.
 */
std::vector<int> TrtDimsToVector(const nvinfer1::Dims& dims) {
  return std::vector<int>(dims.d, dims.d + dims.nbDims);
}

/*!
 * \brief Helper function to convert vector to string.
 * \param vec Vector.
 * \return Vector as a string.
 */
std::string DebugString(const std::vector<int>& vec) {
  std::ostringstream ss;
  ss << "(";
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i != 0) ss << ", ";
    ss << vec[i];
  }
  ss << ")";
  return ss.str();
}

std::vector<int> GetShape(const Type& type) {
  const auto* ttype = type.as<TensorTypeNode>();
  CHECK(ttype);
  std::vector<int> _shape;
  _shape.reserve(ttype->shape.size());
  for (size_t i = 0; i < ttype->shape.size(); ++i) {
    auto* val = ttype->shape[i].as<IntImmNode>();
    _shape.push_back(val ? val->value : -1);
  }
  return _shape;
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TENSORRT_UTILS_H_
