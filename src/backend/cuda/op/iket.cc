/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file backend/cuda/op/iket.cc
 * \brief Frontend-only NVIDIA IKET annotation operators for CUDA.
 */

#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.iket_mark")
      .arg<Expr>("name", "")
      .allow_extra_args()
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.iket."
                                                                      "mark"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.iket_range_start")
      .arg<Expr>("name", "")
      .allow_extra_args()
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.iket."
                                                                      "range_start"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.iket_range_end")
      .arg<Expr>("token", "")
      .allow_extra_args()
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.iket."
                                                                      "range_end"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.iket_range_push")
      .arg<Expr>("name", "")
      .allow_extra_args()
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.iket."
                                                                      "range_push"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.iket_range_pop")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.iket."
                                                                      "range_pop"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.iket_sentinel_token")
      .arg<Expr>("name", "")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.iket."
                                                                      "sentinel_token"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.iket_official_event")
      .arg<Expr>("event_id", "")
      .arg<Expr>("source_code", "")
      .allow_extra_args()
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.iket."
                                                                      "official_event"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace tirx
}  // namespace tvm
