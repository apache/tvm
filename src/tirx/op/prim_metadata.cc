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
 * \file tirx/op/prim_metadata.cc
 * \brief TIRx script metadata for canonical shared primitive operators.
 */
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("prim.likely")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.likely"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.if_then_else")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.if_then_else"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.vscale")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.vscale"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.ceil")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.ceil"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.log2")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.clz")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.clz"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.pow")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.pow"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.fabs")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.fabs"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.fmod")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.fmod"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.floor")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.floor"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.round")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.round"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.nearbyint")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.nearbyint"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.trunc")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.trunc"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.exp")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.exp"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.exp2")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.exp2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.exp10")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.exp10"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.erf")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.erf"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.tanh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tanh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.sigmoid")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sigmoid"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.sqrt")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sqrt"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.rsqrt")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.rsqrt"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.log")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.log1p")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log1p"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.log10")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log10"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.tan")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tan"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.cos")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cos"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.cosh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cosh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.sin")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.sinh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sinh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.asin")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.asin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.acos")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.acos"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.atan")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.atan"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.acosh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.acosh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.asinh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.asinh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.atanh")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.atanh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.atan2")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.atan2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.nextafter")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.nextafter"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.hypot")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.hypot"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.copysign")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.copysign"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.ldexp")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.ldexp"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.isnan")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.isnan"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.popcount")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.popcount"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.fma")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.fma"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));

  OpDef("prim.assume")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.assume"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"));
}

}  // namespace tirx
}  // namespace tvm
