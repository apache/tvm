
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
 * \file tir/op/target_builtin/trn.cc
 *
 *  builtin intrinsic operators specific to Trainium target.
 */
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {
namespace builtin {

#define TIRX_DEFINE_BUILTIN_FUNC(OpName)            \
  const Op& OpName() {                              \
    static const Op& op = Op::Get("tirx." #OpName); \
    return op;                                      \
  }                                                 \
  TVM_TIRX_REGISTER_OP(#OpName)

TIRX_DEFINE_BUILTIN_FUNC(nki_load).set_attr<TCallEffectKind>("TCallEffectKind",
                                                             Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_store).set_attr<TCallEffectKind>("TCallEffectKind",
                                                              Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_tensor_copy)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_matmul)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_activation)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_reciprocal)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_tensortensor)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_tensorscalar)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_memset)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_tensorreduce)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_activation_reduce)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_tensorscalar_reduce)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_identity)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_scalar_tensor_tensor)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_scalar_tensor_scalar)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nki_affine_select)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
