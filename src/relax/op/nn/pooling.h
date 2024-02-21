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
 * \file pooling.h
 * \brief The functions to make Relax neural network pooling operator calls.
 */

#ifndef TVM_RELAX_OP_NN_POOLING_H_
#define TVM_RELAX_OP_NN_POOLING_H_

#include <tvm/relax/attrs/nn.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*! \brief 2D maximum pooling operator. */
Expr max_pool2d(Expr data, Array<IntImm> pool_size, Array<IntImm> strides, Array<IntImm> padding,
                Array<IntImm> dilation, bool ceil_mode, String layout, Optional<String> out_layout);

/*! \brief 2D average pooling operator. */
Expr avg_pool2d(Expr data, Array<IntImm> pool_size, Array<IntImm> strides, Array<IntImm> padding,
                Array<IntImm> dilation, bool ceil_mode, String layout, Optional<String> out_layout);

/*! \brief 2D adaptive average pooling operator. */
Expr adaptive_avg_pool2d(Expr data, Optional<Array<IntImm>> output_size, String layout,
                         Optional<String> out_layout);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_NN_POOLING_H_
