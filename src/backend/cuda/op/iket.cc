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

#define TVM_REGISTER_CUDA_IKET_OP(OpName)                                                    \
  TVM_REGISTER_OP("tirx.cuda.iket_" #OpName)                                                 \
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))              \
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda")) \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.iket." #OpName)) \
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))

TVM_REGISTER_CUDA_IKET_OP(mark);
TVM_REGISTER_CUDA_IKET_OP(range_start);
TVM_REGISTER_CUDA_IKET_OP(range_end);
TVM_REGISTER_CUDA_IKET_OP(range_push);
TVM_REGISTER_CUDA_IKET_OP(range_pop);
TVM_REGISTER_CUDA_IKET_OP(sentinel_token);
TVM_REGISTER_CUDA_IKET_OP(official_event);

#undef TVM_REGISTER_CUDA_IKET_OP

}  // namespace tirx
}  // namespace tvm
