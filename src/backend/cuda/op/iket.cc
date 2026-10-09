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
 * \file backend/cuda/op/iket.cc
 * \brief CUDA IKET annotations.
 */
#include <tvm/backend/cuda/op/iket.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace backend {
namespace cuda {

using namespace tirx;

const Op& iket_mark_op() {
  static const Op op = Op::Get("tirx.cuda.iket_mark");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.iket_mark")
      .signature(sig::arg("name", "The name."), sig::var_args("args"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& iket_range_start_op() {
  static const Op op = Op::Get("tirx.cuda.iket_range_start");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.iket_range_start")
      .signature(sig::arg("name", "The name."), sig::var_args("args"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& iket_range_end_op() {
  static const Op op = Op::Get("tirx.cuda.iket_range_end");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.iket_range_end")
      .signature(sig::arg<IntExpr>("token", "The token."), sig::var_args("args"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& iket_range_push_op() {
  static const Op op = Op::Get("tirx.cuda.iket_range_push");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.iket_range_push")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(sig::arg("name", "The name."), sig::var_args("args"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& iket_range_pop_op() {
  static const Op op = Op::Get("tirx.cuda.iket_range_pop");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.iket_range_pop")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& iket_sentinel_token_op() {
  static const Op op = Op::Get("tirx.cuda.iket_sentinel_token");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.iket_sentinel_token")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32))
      .signature(sig::arg("name", "The name."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.iket_official_event")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32))
      .signature(sig::arg("event_id", "The event identifier."),
                 sig::arg("source_code", "The source code."), sig::var_args("args"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace cuda
}  // namespace backend
}  // namespace tvm
