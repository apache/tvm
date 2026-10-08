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
 * \file tvm/backend/opencl/op.h
 * \brief OpenCL-owned texture operations.
 */
#ifndef TVM_BACKEND_OPENCL_OP_H_
#define TVM_BACKEND_OPENCL_OP_H_

#include <tvm/ir/op.h>

namespace tvm {
namespace backend {
namespace opencl {

/*!
 * \brief Store to texture 2d memory.
 *
 * Arguments, in order:
 * - args[0]: texture, The texture.
 * - args[1]: x, The input value.
 * - args[2]: y, The second input value.
 * - args[3]: z, The third input value.
 * - args[4]: channel_size, The number of channels.
 * - args[5]: value, The value to use.
 */
TVM_DLL const Op& texture2d_store_op();

/*!
 * \brief Load from texture 2d memory.
 *
 * Arguments, in order:
 * - args[0]: texture, The texture.
 * - args[1]: x, The input value.
 * - args[2]: y, The second input value.
 * - args[3]: z, The third input value.
 * - args[4]: channel_size, The number of channels.
 * - args[5]: element_index, The element index within a texture channel.
 */
TVM_DLL const Op& texture2d_load_op();

}  // namespace opencl
}  // namespace backend
}  // namespace tvm

#endif  // TVM_BACKEND_OPENCL_OP_H_
