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
 * \file tirx/op/annotation.cc
 * \brief TIRx annotation operations.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {

const Op& kernel_replace_point_op() {
  static const Op op = Op::Get("tirx.kernel_replace_point");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.kernel_replace_point")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.kernel_replace_point"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& ignore_loop_partition_op() {
  static const Op op = Op::Get("tirx.ignore_loop_partition");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.ignore_loop_partition")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Bool())
      .signature(sig::arg<PrimExpr>("predicate", "The predicate."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.ignore_loop_partition"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

}  // namespace tirx
}  // namespace tvm
