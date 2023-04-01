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
#ifndef TVM_RELAX_OP_VM_H_
#define TVM_RELAX_OP_VM_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {
namespace vm {

// (TVM-TOOL) cc_op begin decl/vm/*
/*!
 * Allocate a storage with specific size and dtype on a specific device.
 *     The allocated storage can be used to create tensors in-place.
 *     The storage is automatically managed by the VM.
 * \param size The shape of the storage.
 * \param runtime_device_index The index of the device on which the storage is allocated.
 * \param dtype The data type of the storage.
 * \return The allocated storage.
 */
relax::Call alloc_storage(relax::Expr size, PrimExpr runtime_device_index, runtime::DataType dtype);
/*!
 * Allocate a tensor with specific shape, dtype on a specific device at the specific offset
 *     on a storage created by R.vm.alloc_storage. The tensor is automatically managed by the VM.
 * \param storage The storage on which the tensor is allocated.
 * \param offset The offset of the tensor on the storage.
 * \param shape The shape of the tensor.
 * \param dtype The data type of the tensor.
 * \return The allocated tensor.
 */
relax::Call alloc_tensor(relax::Expr storage, PrimExpr offset, relax::Expr shape,
                         runtime::DataType dtype);
/*!
 * Call a TIR function with dynamic arguments.
 * \param func The TIR function to be called.
 * \param args The arguments to the TIR function.
 * \return The call node created
 */
relax::Call call_tir_dyn(relax::ExternFunc func, relax::Tuple args);
// (TVM-TOOL) cc_op end decl/vm/*

}  // namespace vm
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_VM_H_
