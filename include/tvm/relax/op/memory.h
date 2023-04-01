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
#ifndef TVM_RELAX_OP_MEMORY_H_
#define TVM_RELAX_OP_MEMORY_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {
namespace memory {

// (TVM-TOOL) cc_op begin decl/memory/*
/*!
 * Allocate a chunk of memory storage with specific size, dtype on a specific device
 *     on its specific storage scope. The allocated storage can be used to create tensors in-place.
 *     The storage will only be freed when the program exits or when the storage is killed by
 *     R.memory.kill_storage.
 * \param size The shape of the storage.
 * \param virtual_device_index The index of the device on which the storage is allocated.
 * \param storage_scope The storage scope of the storage.
 * \param dtype The data type of the storage.
 * \return The allocated storage.
 */
relax::Call alloc_storage(relax::Expr size, PrimExpr virtual_device_index, String storage_scope,
                          runtime::DataType dtype);
/*!
 * Allocate a tensor with specific shape, dtype on a specific device at the specific offset
 *     on a storage created by R.memory.alloc_storage.
 *     The tensor will only be freed when the program exits or when the tensor is killed by
 *     R.memory.kill_tensor.
 * \param storage The storage on which the tensor is allocated.
 * \param offset The offset of the tensor on the storage.
 * \param shape The shape of the tensor.
 * \param dtype The data type of the tensor.
 * \return The allocated tensor.
 */
relax::Call alloc_tensor(relax::Expr storage, PrimExpr offset, relax::Expr shape,
                         runtime::DataType dtype);
/*!
 * Kill a storage created by R.memory.alloc_storage.
 * \param storage The storage being allocated by R.memory.alloc_storage.
 * \return The call node created.
 */
relax::Call kill_storage(relax::Expr storage);
/*!
 * Kill a tensor created by R.memory.alloc_tensor.
 * \param tensor The tensor being allocated by R.memory.alloc_tensor.
 * \return The call node created.
 */
relax::Call kill_tensor(relax::Expr tensor);
// (TVM-TOOL) cc_op end decl/memory/*

}  // namespace memory
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_MEMORY_H_
