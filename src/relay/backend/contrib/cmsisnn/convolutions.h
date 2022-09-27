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
 * \file src/relay/backend/contrib/cmsisnn/convolutions.h
 * \brief CMSIS-NN utility functions for Convolutions
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_CMSISNN_CONVOLUTIONS_H_
#define TVM_RELAY_BACKEND_CONTRIB_CMSISNN_CONVOLUTIONS_H_

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/ndarray.h>

#include "../../../op/make_op.h"
#include "../../../qnn/utils.h"
#include "../../../transforms/pattern_utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {
/*!
 * \brief Checks if Relay Conv2D was originally CMSIS-NN compliant Depthwise Convolution
 * See:
 * https://github.com/apache/tvm/blob/6ed3ab3e33f8eafa4acaf53b7a671831de7587e9/python/tvm/relay/frontend/tflite.py#L2107
 *
 *
 * \return true if a Conv2D is a Depthwise Convolution based on Conv2D's inputs' shapes and
 * attributes
 */

bool IsCMSISNNDepthwise(const Conv2DAttrs* conv2d_attrs, const Array<PrimExpr>& input_shape,
                        const Array<PrimExpr>& kernel_shape);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_CMSISNN_CONVOLUTIONS_H_
