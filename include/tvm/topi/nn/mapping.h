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
 * \brief Mapping op constructions
 * \file nn/mapping.h
 */
#ifndef TVM_TOPI_NN_MAPPING_H_
#define TVM_TOPI_NN_MAPPING_H_

#include <tvm/te/operation.h>
#include <tvm/topi/tags.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Scale and shift with NCHW order
 *
 * \param x The input tensor.
 * \param scale Scale tensor, 1-D of size channel
 * \param shift Shift tensor, 1-D of size channel
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the scale shift operation
 */
inline Tensor scale_shift_nchw(const Tensor& x, const Tensor& scale, const Tensor& shift,
                               std::string name = "ScaleShift", std::string tag = kBroadcast) {
  return tvm::te::compute(
      x->shape, [&](Var b, Var c, Var h, Var w) { return x(b, c, h, w) * scale(c) + shift(c); },
      name, tag);
}

/*!
 * \brief Scale and shift with NHWC order
 *
 * \param x The input tensor.
 * \param scale Scale tensor, 1-D of size channel
 * \param shift Shift tensor, 1-D of size channel
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the scale shift operation
 */
inline Tensor scale_shift_nhwc(const Tensor& x, const Tensor& scale, const Tensor& shift,
                               std::string name = "ScaleShift", std::string tag = kBroadcast) {
  return tvm::te::compute(
      x->shape, [&](Var b, Var h, Var w, Var c) { return x(b, h, w, c) * scale(c) + shift(c); },
      name, tag);
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_NN_MAPPING_H_
