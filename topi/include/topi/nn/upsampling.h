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
 *  Copyright (c) 2017 by Contributors
 * \file topi/nn/upsampling.h
 * \brief upsampling op constructors
 */
#ifndef TOPI_NN_UPSAMPLING_H_
#define TOPI_NN_UPSAMPLING_H_

#include <string>
#include <vector>
#include <iterator>
#include <algorithm>

#include "topi/image/resize.h"

namespace topi {
namespace nn {
using namespace tvm;
using namespace topi::image;

/*!
* \brief Upsample given tensor to given shape
*
* \param input The input tensor.
* \param shape Output shape to upsample.
* \param layout input layout
* \param mode Algorithm to use (NEAREST_NEIGHBOR / BILINEAR)
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor upsampled to given shape
*/
inline Tensor upsampling(const Tensor& input,
                         const Array<Expr> shape,
                         std::string layout = "NCHW",
                         std::string mode = "NEAREST_NEIGHBOR",
                         std::string name = "tensor",
                         std::string tag = kInjective) {
  return resize(input, shape, layout, false, mode);
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_UPSAMPLING_H_
