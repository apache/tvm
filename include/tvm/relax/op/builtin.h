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
#ifndef TVM_RELAX_OP_BUILTIN_H_
#define TVM_RELAX_OP_BUILTIN_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {
namespace builtin {

// (TVM-TOOL) cc_op begin decl/builtin/*
/*!
 * Construct a Call to allocate a tensor with specific shape, dtype, and the index of
 *     the device it is constructed on.
 * \param shape The shape of the tensor.
 * \param dtype The data type of the tensor.
 * \param runtime_device_index The index of the device it is constructed on.
 * \return The created call node.
 */
relax::Call alloc_tensor(relax::Expr shape, runtime::DataType dtype, int64_t runtime_device_index);
/*!
 * An indicator op that the consumers of input tensor should not be
 *     lifted to transform_params function.
 * \param x The input tensor.
 * \return The created call node.
 */
relax::Call stop_lift_params(relax::Expr x);
// (TVM-TOOL) cc_op end decl/builtin/*

}  // namespace builtin
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_BUILTIN_H_
