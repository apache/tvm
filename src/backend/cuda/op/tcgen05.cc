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

#include <tvm/backend/cuda/op/tcgen05.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace backend {
namespace cuda {

using namespace tirx;

void TCGen05InstrDescriptorAttrsNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<TCGen05InstrDescriptorAttrsNode>()
      .def_ro("d_dtype", &TCGen05InstrDescriptorAttrsNode::d_dtype)
      .def_ro("a_dtype", &TCGen05InstrDescriptorAttrsNode::a_dtype)
      .def_ro("b_dtype", &TCGen05InstrDescriptorAttrsNode::b_dtype)
      .def_ro("M", &TCGen05InstrDescriptorAttrsNode::M)
      .def_ro("N", &TCGen05InstrDescriptorAttrsNode::N)
      .def_ro("K", &TCGen05InstrDescriptorAttrsNode::K)
      .def_ro("trans_a", &TCGen05InstrDescriptorAttrsNode::trans_a)
      .def_ro("trans_b", &TCGen05InstrDescriptorAttrsNode::trans_b)
      .def_ro("n_cta_groups", &TCGen05InstrDescriptorAttrsNode::n_cta_groups, refl::DefaultValue(1))
      .def_ro("neg_a", &TCGen05InstrDescriptorAttrsNode::neg_a, refl::DefaultValue(false))
      .def_ro("neg_b", &TCGen05InstrDescriptorAttrsNode::neg_b, refl::DefaultValue(false))
      .def_ro("sat_d", &TCGen05InstrDescriptorAttrsNode::sat_d, refl::DefaultValue(false))
      .def_ro("is_sparse", &TCGen05InstrDescriptorAttrsNode::is_sparse, refl::DefaultValue(false));
}

void TCGen05InstrDescriptorBlockScaledAttrsNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<TCGen05InstrDescriptorBlockScaledAttrsNode>()
      .def_ro("d_dtype", &TCGen05InstrDescriptorBlockScaledAttrsNode::d_dtype)
      .def_ro("a_dtype", &TCGen05InstrDescriptorBlockScaledAttrsNode::a_dtype)
      .def_ro("b_dtype", &TCGen05InstrDescriptorBlockScaledAttrsNode::b_dtype)
      .def_ro("sfa_dtype", &TCGen05InstrDescriptorBlockScaledAttrsNode::sfa_dtype)
      .def_ro("sfb_dtype", &TCGen05InstrDescriptorBlockScaledAttrsNode::sfb_dtype)
      .def_ro("M", &TCGen05InstrDescriptorBlockScaledAttrsNode::M)
      .def_ro("N", &TCGen05InstrDescriptorBlockScaledAttrsNode::N)
      .def_ro("K", &TCGen05InstrDescriptorBlockScaledAttrsNode::K)
      .def_ro("trans_a", &TCGen05InstrDescriptorBlockScaledAttrsNode::trans_a)
      .def_ro("trans_b", &TCGen05InstrDescriptorBlockScaledAttrsNode::trans_b)
      .def_ro("n_cta_groups", &TCGen05InstrDescriptorBlockScaledAttrsNode::n_cta_groups,
              refl::DefaultValue(1))
      .def_ro("neg_a", &TCGen05InstrDescriptorBlockScaledAttrsNode::neg_a,
              refl::DefaultValue(false))
      .def_ro("neg_b", &TCGen05InstrDescriptorBlockScaledAttrsNode::neg_b,
              refl::DefaultValue(false))
      .def_ro("is_sparse", &TCGen05InstrDescriptorBlockScaledAttrsNode::is_sparse,
              refl::DefaultValue(false));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  TCGen05InstrDescriptorAttrsNode::RegisterReflection();
  TCGen05InstrDescriptorBlockScaledAttrsNode::RegisterReflection();

  OpDef("tirx.cuda.tcgen05_encode_instr_descriptor")
      .signature(sig::arg("desc"),
                 sig::call_attrs<backend::cuda::TCGen05InstrDescriptorAttrsNode>())
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>(tvm::tirx::op_attr::kDeviceIntrinsicNamespace,
                                           ffi::String("cuda"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  OpDef("tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled")
      .signature(sig::arg("desc"),
                 sig::call_attrs<backend::cuda::TCGen05InstrDescriptorBlockScaledAttrsNode>())
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>(tvm::tirx::op_attr::kDeviceIntrinsicNamespace,
                                           ffi::String("cuda"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  OpDef("tirx.cuda.tcgen05_encode_matrix_descriptor")
      .signature(sig::arg("desc"), sig::arg("addr"), sig::arg<IntExpr>("ldo"),
                 sig::arg<IntExpr>("sdo"), sig::arg<IntExpr>("swizzle"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>(tvm::tirx::op_attr::kDeviceIntrinsicNamespace,
                                           ffi::String("cuda"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
}

}  // namespace cuda
}  // namespace backend
}  // namespace tvm
