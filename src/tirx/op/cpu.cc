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
 * \file tirx/op/cpu.cc
 * \brief TIRx cpu operations.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/tirx/op/cpu.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {

const Op& cpu_parallel_barrier_op() {
  static const Op op = Op::Get("tirx.cpu_parallel_barrier");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cpu_parallel_barrier")
      .signature()
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.cpu_parallel_barrier"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace tirx
}  // namespace tvm
