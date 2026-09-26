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
#include <tvm/runtime/base.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {
namespace builtin {

void RegisterMetalTargetBuiltins() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  OpDef("tirx.make_filled_simdgroup_matrix")
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("metal.make_filled_simdgroup_matrix"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.simdgroup_load")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("metal.simdgroup_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.simdgroup_store")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("metal.simdgroup_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.simdgroup_multiply_accumulate")
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("metal.simdgroup_multiply_accumulate"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

TVM_FFI_STATIC_INIT_BLOCK() { RegisterMetalTargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
