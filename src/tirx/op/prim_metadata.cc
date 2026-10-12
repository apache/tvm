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
#include <tvm/script/printer/doc_translator.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("prim.likely")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.likely"));

  OpDef("prim.if_then_else")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.if_then_else"));

  OpDef("prim.vscale")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.vscale"));

  OpDef("prim.ceil")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.ceil"));

  OpDef("prim.log2")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.log2"));

  OpDef("prim.clz")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.clz"));

  OpDef("prim.pow")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.pow"));

  OpDef("prim.fabs")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.fabs"));

  OpDef("prim.fmod")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.fmod"));

  OpDef("prim.floor")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.floor"));

  OpDef("prim.round")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.round"));

  OpDef("prim.nearbyint")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.nearbyint"));

  OpDef("prim.trunc")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.trunc"));

  OpDef("prim.exp")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.exp"));

  OpDef("prim.exp2")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.exp2"));

  OpDef("prim.exp10")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.exp10"));

  OpDef("prim.erf")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.erf"));

  OpDef("prim.tanh")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.tanh"));

  OpDef("prim.sigmoid")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.sigmoid"));

  OpDef("prim.sqrt")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.sqrt"));

  OpDef("prim.rsqrt")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.rsqrt"));

  OpDef("prim.log")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.log"));

  OpDef("prim.log1p")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.log1p"));

  OpDef("prim.log10")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.log10"));

  OpDef("prim.tan")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.tan"));

  OpDef("prim.cos")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.cos"));

  OpDef("prim.cosh")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.cosh"));

  OpDef("prim.sin")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.sin"));

  OpDef("prim.sinh")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.sinh"));

  OpDef("prim.asin")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.asin"));

  OpDef("prim.acos")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.acos"));

  OpDef("prim.atan")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.atan"));

  OpDef("prim.acosh")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.acosh"));

  OpDef("prim.asinh")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.asinh"));

  OpDef("prim.atanh")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.atanh"));

  OpDef("prim.atan2")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.atan2"));

  OpDef("prim.nextafter")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.nextafter"));

  OpDef("prim.hypot")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.hypot"));

  OpDef("prim.copysign")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.copysign"));

  OpDef("prim.ldexp")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.ldexp"));

  OpDef("prim.isnan")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.isnan"));

  OpDef("prim.popcount")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.popcount"));

  OpDef("prim.fma")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.fma"));

  OpDef("prim.assume")
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.assume"));
}

}  // namespace tirx
}  // namespace tvm
