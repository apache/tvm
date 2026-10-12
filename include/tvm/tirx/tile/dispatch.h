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
/*! \brief Dispatch interface for lowering tile instruction Calls. */
#ifndef TVM_TIRX_TILE_DISPATCH_H_
#define TVM_TIRX_TILE_DISPATCH_H_

#include <tvm/ffi/function.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/tile/context.h>

namespace tvm::tirx::tile {

/*!
 * \brief Lower a tile instruction Call using its execution context.
 *
 * Registering this attribute identifies an Op as a tile instruction. The
 * dispatcher can be implemented in C++ or registered from Python.
 */
using FTileOpDispatch = ffi::TypedFunction<Function(const Call&, const DispatchContext&)>;

namespace op_attr {
inline constexpr const char* kTileOpDispatch = "FTileOpDispatch";
}  // namespace op_attr

/*!
 * \brief Invoke the Call's FTileOpDispatch attribute.
 * \param call The tile instruction to lower.
 * \param context The operator's execution context and accumulated callbacks.
 * \return The instruction implementation, whose body is expanded by TileDispatch.
 */
TVM_TIRX_DLL Function Dispatch(const Call& call, const DispatchContext& context);

}  // namespace tvm::tirx::tile
#endif  // TVM_TIRX_TILE_DISPATCH_H_
