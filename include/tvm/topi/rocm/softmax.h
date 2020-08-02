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
 * \file rocm/injective.h
 * \brief ROCM schedule for injective operations
 */
#ifndef TVM_TOPI_ROCM_SOFTMAX_H_
#define TVM_TOPI_ROCM_SOFTMAX_H_

#include <tvm/target/generic_func.h>
#include <tvm/te/operation.h>
#include <tvm/topi/cuda/softmax.h>
#include <tvm/topi/detail/fuse.h>
#include <tvm/topi/tags.h>

namespace tvm {
namespace topi {

using namespace tvm::te;

namespace rocm {

/*!
 * \brief Create a rocm schedule for the given softmax output tensors.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_softmax(const Target& target, const Array<Tensor>& outs) {
  return topi::cuda::schedule_softmax(target, outs);
}

}  // namespace rocm
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_ROCM_SOFTMAX_H_
