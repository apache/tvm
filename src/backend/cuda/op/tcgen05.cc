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
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace backend {
namespace cuda {

using namespace tirx;

void TCGen05InstrDescriptorAttrs::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<TCGen05InstrDescriptorAttrs>()
      .def_ro("d_dtype", &TCGen05InstrDescriptorAttrs::d_dtype)
      .def_ro("a_dtype", &TCGen05InstrDescriptorAttrs::a_dtype)
      .def_ro("b_dtype", &TCGen05InstrDescriptorAttrs::b_dtype)
      .def_ro("M", &TCGen05InstrDescriptorAttrs::M)
      .def_ro("N", &TCGen05InstrDescriptorAttrs::N)
      .def_ro("K", &TCGen05InstrDescriptorAttrs::K)
      .def_ro("trans_a", &TCGen05InstrDescriptorAttrs::trans_a)
      .def_ro("trans_b", &TCGen05InstrDescriptorAttrs::trans_b)
      .def_ro("n_cta_groups", &TCGen05InstrDescriptorAttrs::n_cta_groups, refl::DefaultValue(1))
      .def_ro("neg_a", &TCGen05InstrDescriptorAttrs::neg_a, refl::DefaultValue(false))
      .def_ro("neg_b", &TCGen05InstrDescriptorAttrs::neg_b, refl::DefaultValue(false))
      .def_ro("sat_d", &TCGen05InstrDescriptorAttrs::sat_d, refl::DefaultValue(false))
      .def_ro("is_sparse", &TCGen05InstrDescriptorAttrs::is_sparse, refl::DefaultValue(false));
}

void TCGen05InstrDescriptorBlockScaledAttrs::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<TCGen05InstrDescriptorBlockScaledAttrs>()
      .def_ro("d_dtype", &TCGen05InstrDescriptorBlockScaledAttrs::d_dtype)
      .def_ro("a_dtype", &TCGen05InstrDescriptorBlockScaledAttrs::a_dtype)
      .def_ro("b_dtype", &TCGen05InstrDescriptorBlockScaledAttrs::b_dtype)
      .def_ro("sfa_dtype", &TCGen05InstrDescriptorBlockScaledAttrs::sfa_dtype)
      .def_ro("sfb_dtype", &TCGen05InstrDescriptorBlockScaledAttrs::sfb_dtype)
      .def_ro("M", &TCGen05InstrDescriptorBlockScaledAttrs::M)
      .def_ro("N", &TCGen05InstrDescriptorBlockScaledAttrs::N)
      .def_ro("K", &TCGen05InstrDescriptorBlockScaledAttrs::K)
      .def_ro("trans_a", &TCGen05InstrDescriptorBlockScaledAttrs::trans_a)
      .def_ro("trans_b", &TCGen05InstrDescriptorBlockScaledAttrs::trans_b)
      .def_ro("n_cta_groups", &TCGen05InstrDescriptorBlockScaledAttrs::n_cta_groups,
              refl::DefaultValue(1))
      .def_ro("neg_a", &TCGen05InstrDescriptorBlockScaledAttrs::neg_a, refl::DefaultValue(false))
      .def_ro("neg_b", &TCGen05InstrDescriptorBlockScaledAttrs::neg_b, refl::DefaultValue(false))
      .def_ro("is_sparse", &TCGen05InstrDescriptorBlockScaledAttrs::is_sparse,
              refl::DefaultValue(false));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  TCGen05InstrDescriptorAttrs::RegisterReflection();
  TCGen05InstrDescriptorBlockScaledAttrs::RegisterReflection();

  OpDef("tirx.cuda.tcgen05_encode_instr_descriptor")
      .signature(sig::arg("desc"), sig::call_attrs<backend::cuda::TCGen05InstrDescriptorAttrs>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  OpDef("tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled")
      .signature(sig::arg("desc"),
                 sig::call_attrs<backend::cuda::TCGen05InstrDescriptorBlockScaledAttrs>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  OpDef("tirx.cuda.tcgen05_encode_matrix_descriptor")
      .signature(sig::arg("desc"), sig::arg("addr"), sig::arg<IntExpr>("ldo"),
                 sig::arg<IntExpr>("sdo"), sig::arg<IntExpr>("swizzle"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
}

}  // namespace cuda
}  // namespace backend
}  // namespace tvm
