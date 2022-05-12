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
 * \file tir/analysis/device_constraint_utils.cc
 * \brief Utilities for extracting and applying device-related constraints to \p PrimFunc
 * parameters.
 *
 * These utilities are used by the \p PlanDevices pass to extract memory (aka 'storage') scope
 * information from \p PrimFuncs and convert them back into \p VirtualDevice form w.r.t. the
 * original Relay type of the \p PrimFunc (ie before flattening of tuple arguments/results and
 * conversion to destination-passing style aka DPS).
 *
 * A utility is also supplied to go the other way: impose memory scopes on \p PrimFunc parameters.
 * However that's still in EXPERIMENTAL form.
 *
 * We may extend these utilities to also gather/apply layout information should we add that to
 * \p VirtualDevice.
 */

#ifndef TVM_TIR_ANALYSIS_DEVICE_CONSTRAINT_UTILS_H_
#define TVM_TIR_ANALYSIS_DEVICE_CONSTRAINT_UTILS_H_

#include <tvm/target/virtual_device.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace tir {

/*!
 * A Relay Function with type:
 * \code
 *   fn((Tensor[...], Tensor[...]), Tensor[...]) -> (Tensor[...], Tensor[...])
 *       ^            ^             ^                ^            ^
 *       a            b             c                d            e
 * \endcode
 * will be represented by a TIR PrimFunc in flattened and DPS form with at least 5 argument a..e.
 * \code
 *   primfn(a: handle, b: handle, c: handle, d: handle, e: handle) {
 *     buffers = { ... }
 *     buffer_map = { ... }
 *     ...
 *   }
 * \endcode
 *
 * Each such PrimFunc argument will me mapped to a \p Buffer who's underlying \p data \p Var
 * has a \p PointerType.
 *
 * The PrimFunc may have additional non-pointer arguments, eg for:
 *  - scalar inputs and tensor dimensions
 *  - device contexts
 * Those should be ignored here since they have no counterpart in the Relay Function.
 *
 * We'll need helpers to map on-the-fly between the Relay and TIR view of functions.
 */

/*!
 * \brief Returns the \p VirtualDevices capturing the memory (aka storage) scope constraints for all
 * the arguments and result of \p prim_func. However the result will be w.r.t. the \p prim_func's
 * representation as a Relay \p Function of \p relay_func_type_ before lowering and conversion to
 * DPS.
 */
Array<VirtualDevice> GetPrimFuncArgAndResultConstraints(const tir::PrimFunc& prim_func,
                                                        const FuncType& relay_func_type);

/*
 * \brief Returns \p prim_func written to capture the memory (aka storage) scope constraints
 * for each of the \p prim_func's parameters given by \p arg_and_result_virtual_devices. However,
 * \p arg_and_result_virtual_devices should be w.r.t. the \p prim_func's representation as a Relay
 * \p Function of \p relay_func_type before lowering and conversion to DPS.
 *
 * CAUTION: This is experimental. The resulting \p PrimFunc may not have fully accounted for all
 * new memory scopes.
 */
PrimFunc ApplyPrimFuncArgAndResultConstraints(
    const PrimFunc& prim_func, const FuncType& relay_func_type,
    const Array<VirtualDevice>& arg_and_result_virtual_devices);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_ANALYSIS_DEVICE_CONSTRAINT_UTILS_H_
