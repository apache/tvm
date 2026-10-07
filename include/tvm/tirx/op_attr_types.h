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
 * \file tvm/tirx/op_attr_types.h
 * \brief Attribute types in the Op registry for TIR ops.
 *
 * These attributes can be set via OpDef::set_attr
 *
 * \sa tvm/ir/op.h
 */
#ifndef TVM_TIR_OP_ATTR_TYPES_H_
#define TVM_TIR_OP_ATTR_TYPES_H_

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/native_function.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>

namespace tvm {
namespace tirx {
/*!
 * \brief Construct fresh typed variables for a region's lexical body parameters.
 *
 * The input carries only the operation, operands and attributes, without a body
 * or builder state. Parameters are ordered, distinct definitions and may have
 * name hints. Every region operation must register this hook, returning an empty
 * array when it has no body parameters. Presence of the attribute identifies
 * region support without invoking the hook or allocating variables.
 */
using FRegionGetBodyParams =
    ffi::reflection::NativeFunctionView<ffi::Array<Var>(const CallNode* call)>;

/*! \brief Shared FRegionGetBodyParams implementation for regions without body parameters. */
inline ffi::Array<Var> RegionNoBodyParams(const CallNode*) { return {}; }

/*!
 * \brief Global symbol of the op after lowering.
 */
using TGlobalSymbol = ffi::String;

/*!
 * \brief Whether the op is overloaded for vector form.
 */
using TVectorizable = bool;

/*!
 * \brief The intrinsic lowering function for given op.
 */
using FLowerIntrinsic = ffi::TypedFunction<PrimExpr(PrimExpr)>;

/*!
 * \brief The legalization function for given tirx op.
 */
using FLegalize = ffi::TypedFunction<PrimExpr(PrimExpr)>;

/*!
 * \brief The fully qualified TVMScript name, including its dialect namespace.
 */
using TScriptPrinterName = ffi::String;

/*!
 * \brief Specifies that TVMScript printer prints the dtype as the first/last argument.
          If not specified, dtype will not be printed.
 */
enum class ScriptDtypePrintLocation : int {
  /*!
   * \brief Do not print dtype as an argument.
   */
  kNone = 0,
  /*!
   * \brief Print dtype as the first argument.
   */
  kFirst = 1,
  /*!
   * \brief FPrint dtype as the last argument.
   */
  kLast = 2,
};

using TScriptDtypePrintLocation = int64_t;

/*!
 * \brief Broad TIRx op category.
 *
 * Expected values:
 * - "builtin"
 * - "tile_primitive"
 * - "device_intrin"
 */
using TIRxOpCategory = ffi::String;

/*!
 * \brief Device intrinsic namespace.
 *
 * Expected values include "cuda", "ptx", "nvshmem", "nki", and "metal".
 */
using TDeviceIntrinsicNamespace = ffi::String;

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_OP_ATTR_TYPES_H_
