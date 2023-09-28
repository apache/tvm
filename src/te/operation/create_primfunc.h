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

#ifndef TVM_TE_OPERATION_CREATE_PRIMFUNC_H_
#define TVM_TE_OPERATION_CREATE_PRIMFUNC_H_

#include <tvm/runtime/container/array.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/function.h>

#include <optional>

namespace tvm {
namespace tir {

/*! \brief Use Tensor Expression to create a schedulable TensorIR func. */
PrimFunc CreatePrimFunc(const Array<te::Tensor>& arg_list,
                        std::optional<DataType> index_dtype_override = std::nullopt);

/*! \brief The same as above but create a PrimFunc with AllocateConstNode. If the size of the
 * constants array is N, the last N tensors in arg_list will be treated as constant tensors.
 * Constant tensors will not be part of the parameters of the created PrimFunc, instead constants
 * will be embedded in the body as AllocateConstNode.
 */
PrimFunc CreatePrimFuncWithConstants(const Array<te::Tensor>& arg_list,
                                     const Array<runtime::NDArray>& constants,
                                     std::optional<DataType> index_dtype_override = std::nullopt);

// Relax version
// TODO(relax-team) combine with the relay version
/*! \brief Use Tensor Expression to create a schedulable TensorIR func. */
PrimFunc CreatePrimFunc(const Array<ObjectRef>& arg_list,
                        std::optional<DataType> index_dtype_override);

/*! \brief The same as above but create a PrimFunc with AllocateConstNode. If the size of the
 * constants array is N, the last N tensors in arg_list will be treated as constant tensors.
 * Constant tensors will not be part of the parameters of the created PrimFunc, instead constants
 * will be embedded in the body as AllocateConstNode.
 */
PrimFunc CreatePrimFuncWithConstants(const Array<ObjectRef>& arg_list,
                                     const Array<runtime::NDArray>& constants,
                                     std::optional<DataType> index_dtype_override);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TE_OPERATION_CREATE_PRIMFUNC_H_
