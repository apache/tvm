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
#include "convolutions.h"

#include <string>

#include "../../../qnn/utils.h"
#include "tvm/ir/transform.h"
#include "tvm/relay/attrs/nn.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

bool IsCMSISNNDepthwise(const Conv2DAttrs* conv2d_attrs, const Array<PrimExpr>& input_shape,
                        const Array<PrimExpr>& kernel_shape) {
  std::string kernel_layout = conv2d_attrs->kernel_layout.c_str();
  int kernel_pos_o = kernel_layout.find("O");
  int kernel_pos_i = kernel_layout.find("I");
  int kernel_dim_o_val = qnn::get_const_int(kernel_shape[kernel_pos_o]);
  int kernel_dim_i_val = qnn::get_const_int(kernel_shape[kernel_pos_i]);
  int64_t out_channels = conv2d_attrs->channels.as<IntImmNode>()->value;
  return (out_channels == kernel_dim_o_val * kernel_dim_i_val);
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
