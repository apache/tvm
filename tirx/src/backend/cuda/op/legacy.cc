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
 * \file backend/cuda/op/legacy.cc
 * \brief Legacy CUDA operations retained for downstream code generators.
 */
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.mma_store")
      .signature(sig::arg<IntExpr>("m", "The M dimension."),
                 sig::arg<IntExpr>("n", "The N dimension."),
                 sig::arg("dst_ptr", "The destination pointer."),
                 sig::arg("src_ptr", "The source pointer."),
                 sig::arg<IntExpr>("src_offset", "The source offset."),
                 sig::arg<IntExpr>("dst_stride", "The destination stride."))
      .set_attr(tvm::op_attr::kCallEffectKind, static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.mma_fill")
      .signature(sig::arg<IntExpr>("local_size", "The local allocation size."),
                 sig::arg("local_ptr", "The local pointer."),
                 sig::arg<IntExpr>("offset", "The offset."))
      .set_attr(tvm::op_attr::kCallEffectKind, static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.ptx_legacy.ldmatrix")
      .signature(sig::arg("trans"), sig::arg<IntExpr>("num"), sig::arg("dtype"),
                 sig::arg("local_ptr"), sig::arg<IntExpr>("local_offset"), sig::arg("smem_ptr"),
                 sig::arg<IntExpr>("smem_offset"))
      .set_attr(tvm::op_attr::kCallEffectKind, static_cast<int64_t>(CallEffectKind::kOpaque));
  OpDef("tirx.ptx_legacy.mma")
      .signature(sig::arg("shape"), sig::arg("a_layout"), sig::arg("b_layout"), sig::arg("a_dtype"),
                 sig::arg("b_dtype"), sig::arg("c_dtype"), sig::arg("a_ptr"),
                 sig::arg<IntExpr>("a_offset"), sig::arg("b_ptr"), sig::arg<IntExpr>("b_offset"),
                 sig::arg("acc_ptr"), sig::arg<IntExpr>("c_offset"), sig::arg("saturate"),
                 sig::var_args("args"))
      .set_attr(tvm::op_attr::kCallEffectKind, static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace tirx
}  // namespace tvm
