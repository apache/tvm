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
 * \brief Reorg op constructions
 * \file vision/reorg.h
 */
#ifndef TVM_TOPI_VISION_REORG_H_
#define TVM_TOPI_VISION_REORG_H_

#include <tvm/te/operation.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/tags.h>
#include <tvm/topi/transform.h>

#include <algorithm>
#include <string>

namespace tvm {
namespace topi {
namespace vision {

using namespace tvm::te;

/*!
 * \brief Reorg operation
 *
 * \param data The input tensor. Can be any dimension
 * \param stride The input integer used as stride in reorg operation
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the reorg operation
 */
inline Tensor reorg(const Tensor& data, int stride = 1, std::string name = "tensor",
                    std::string tag = "reorg_output") {
  auto input_shape = data->shape;

  int batch = GetConstInt(input_shape[0]);
  int c_in = GetConstInt(input_shape[1]);
  int h_in = GetConstInt(input_shape[2]);
  int w_in = GetConstInt(input_shape[3]);
  int out_c = c_in / (stride * stride);

  auto out = tvm::te::compute(
      input_shape,
      [&](Var b, Var k, Var j, Var i) {
        return data(b * stride * stride, indexmod(k, out_c) * stride * stride,
                    (j * stride + indexdiv(indexdiv(k, out_c), stride)) * stride,
                    (i * stride + indexmod(indexdiv(k, out_c), stride)));
      },
      name, tag);

  out_c = c_in * stride * stride;
  int out_h = h_in / stride;
  int out_w = w_in / stride;

  Array<PrimExpr> out_shape = {batch, out_c, out_h, out_w};
  return reshape(out, out_shape);
}
}  // namespace vision
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_VISION_REORG_H_
