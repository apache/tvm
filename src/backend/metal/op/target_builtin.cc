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
 * \file backend/metal/op/target_builtin.cc
 *
 *  builtin intrinsic operators specific to Metal target.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/base.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {
namespace builtin {

void RegisterMetalTargetBuiltins() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  OpDef("tirx.metal.make_filled_simdgroup_matrix")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(sig::arg("d", "The D operand."), sig::arg<IntExpr>("index", "The index."),
                 sig::arg("value", "The value to use."),
                 sig::arg<IntExpr>("col", "The column index."),
                 sig::arg<IntExpr>("row", "The row index."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.metal.make_filled_simdgroup_matrix"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.metal.simdgroup_load")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(sig::arg("d", "The D operand."), sig::arg<IntExpr>("index", "The index."),
                 sig::arg("ptr", "The pointer."), sig::arg<IntExpr>("stride", "The stride."),
                 sig::arg<IntExpr>("col", "The column index."),
                 sig::arg<IntExpr>("row", "The row index."),
                 sig::arg("transpose_matrix", "Whether to transpose the matrix."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.metal.simdgroup_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.metal.simdgroup_store")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(sig::arg("d", "The D operand."), sig::arg<IntExpr>("index", "The index."),
                 sig::arg("ptr", "The pointer."), sig::arg<IntExpr>("stride", "The stride."),
                 sig::arg<IntExpr>("col", "The column index."),
                 sig::arg<IntExpr>("row", "The row index."),
                 sig::arg("transpose_matrix", "Whether to transpose the matrix."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.metal.simdgroup_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.metal.simdgroup_multiply_accumulate")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(
          sig::arg("d", "The D operand."), sig::arg<IntExpr>("index_d", "The D fragment index."),
          sig::arg("a", "The A operand."), sig::arg<IntExpr>("index_a", "The A fragment index."),
          sig::arg("b", "The B operand."), sig::arg<IntExpr>("index_b", "The B fragment index."),
          sig::arg("c", "The C operand."), sig::arg<IntExpr>("index_c", "The C fragment index."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.metal.simdgroup_multiply_accumulate"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

TVM_FFI_STATIC_INIT_BLOCK() { RegisterMetalTargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
