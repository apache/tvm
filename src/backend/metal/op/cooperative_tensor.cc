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

#include <tvm/backend/metal/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm::backend::metal {

using namespace tirx;

const Op& cooperative_tensor_fill_op() {
  static const Op op = Op::Get("tirx.metal.cooperative_tensor_fill");
  return op;
}
const Op& cooperative_tensor_load_op() {
  static const Op op = Op::Get("tirx.metal.cooperative_tensor_load");
  return op;
}
const Op& cooperative_tensor_store_op() {
  static const Op op = Op::Get("tirx.metal.cooperative_tensor_store");
  return op;
}
const Op& cooperative_tensor_multiply_accumulate_op() {
  static const Op op = Op::Get("tirx.metal.cooperative_tensor_multiply_accumulate");
  return op;
}
TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.metal.cooperative_tensor_fill")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(sig::arg("d", "The D operand."), sig::arg<IntExpr>("index", "The index."),
                 sig::arg<PrimExpr>("value", "The value to use."),
                 sig::arg<IntExpr>("rows", "The number of rows."),
                 sig::arg<IntExpr>("cols", "The number of columns."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.metal.cooperative_tensor_fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
  OpDef("tirx.metal.cooperative_tensor_load")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(sig::arg("d", "The D operand."), sig::arg<IntExpr>("index", "The index."),
                 sig::arg("ptr", "The pointer."), sig::arg<IntExpr>("stride", "The stride."),
                 sig::arg<IntExpr>("rows", "The number of rows."),
                 sig::arg<IntExpr>("cols", "The number of columns."),
                 sig::arg<PrimExpr>("transpose_matrix", "Whether to transpose the matrix."),
                 sig::arg<IntExpr>("mma_M", "The M dimension of the matrix operation."),
                 sig::arg<IntExpr>("mma_N", "The N dimension of the matrix operation."),
                 sig::arg<IntExpr>("mma_K", "The K dimension of the matrix operation."),
                 sig::arg<IntExpr>("operand_role", "The matrix operand role."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.metal.cooperative_tensor_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
  OpDef("tirx.metal.cooperative_tensor_store")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(sig::arg("d", "The D operand."), sig::arg<IntExpr>("index", "The index."),
                 sig::arg("ptr", "The pointer."), sig::arg<IntExpr>("stride", "The stride."),
                 sig::arg<IntExpr>("rows", "The number of rows."),
                 sig::arg<IntExpr>("cols", "The number of columns."),
                 sig::arg<PrimExpr>("transpose_matrix", "Whether to transpose the matrix."),
                 sig::arg<IntExpr>("mma_M", "The M dimension of the matrix operation."),
                 sig::arg<IntExpr>("mma_N", "The N dimension of the matrix operation."),
                 sig::arg<IntExpr>("mma_K", "The K dimension of the matrix operation."),
                 sig::arg<IntExpr>("operand_role", "The matrix operand role."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.metal.cooperative_tensor_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
  OpDef("tirx.metal.cooperative_tensor_multiply_accumulate")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(
          sig::arg("d", "The D operand."), sig::arg<IntExpr>("index_d", "The D fragment index."),
          sig::arg("a", "The A operand."), sig::arg<IntExpr>("index_a", "The A fragment index."),
          sig::arg("b", "The B operand."), sig::arg<IntExpr>("index_b", "The B fragment index."),
          sig::arg("c", "The C operand."), sig::arg<IntExpr>("index_c", "The C fragment index."),
          sig::arg<IntExpr>("M", "The M dimension."), sig::arg<IntExpr>("N", "The N dimension."),
          sig::arg<IntExpr>("K", "The K dimension."),
          sig::arg<PrimExpr>("transpose_a", "Whether to transpose A."),
          sig::arg<PrimExpr>("transpose_b", "Whether to transpose B."))
      .set_attr<TScriptPrinterName>(
          "TScriptPrinterName", ffi::String("tirx.metal.cooperative_tensor_multiply_accumulate"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace tvm::backend::metal
