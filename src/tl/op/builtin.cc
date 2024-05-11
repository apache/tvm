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
 * \file tl/op/builtin.cc
 * \brief Builtin intrinsics.
 *
 */

#include "builtin.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/utils.h"
#include "cuda.h"

namespace tvm {
namespace tl {

#define TIR_DEFINE_TL_BUILTIN(OpName)             \
  const Op& OpName() {                            \
    static const Op& op = Op::Get("tl." #OpName); \
    return op;                                    \
  }                                               \
  TVM_REGISTER_OP("tl." #OpName).set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)

TIR_DEFINE_TL_BUILTIN(CreateListofMBarrierOp)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(CreateTMADescriptorOp)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(CreateTMAIm2ColDescriptorOp)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(GetMBarrierOp)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(TMALoadOp).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(TMALoadIm2ColOp).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    
TIR_DEFINE_TL_BUILTIN(TMAStoreOp)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(MBarrierWaitParity)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(MBarrierExpectTX)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(LDMatrixOp)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(STMatrixOp)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(SyncThreadsPartialOp)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(FenceProxyAsyncOp)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(PackB16Op).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));
}  // namespace tl
}  // namespace tvm