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
/*! \brief Operator registration contracts for tile instructions. */
#ifndef TVM_TIRX_TILE_OP_H_
#define TVM_TIRX_TILE_OP_H_

#include <tvm/ffi/function.h>
#include <tvm/ir/op.h>

namespace tvm::tirx::tile {

namespace op_attr {
/*! \brief Backend validation of a tensor instruction Call. */
inline constexpr const char* kTensorCallValidate = "FTensorCallValidate";
}  // namespace op_attr

/*! \brief Attach the native Call contract and backend validator to a tile instruction. */
TVM_DLL void ConfigureTensorInstruction(Op op, ffi::Function validator);

}  // namespace tvm::tirx::tile
#endif  // TVM_TIRX_TILE_OP_H_
