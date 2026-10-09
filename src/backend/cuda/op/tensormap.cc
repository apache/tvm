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

#include <tvm/backend/cuda/op/tensormap.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace backend {
namespace cuda {

using namespace tirx;

void TensorMapEncodeTiledAttr::RegisterReflection() {
  namespace refl = ffi::reflection;
  ffi::reflection::ObjectDef<TensorMapEncodeTiledAttr>()
      .def_ro("descriptor_dtype", &TensorMapEncodeTiledAttr::descriptor_dtype)
      .def_ro("rank", &TensorMapEncodeTiledAttr::rank)
      .def_ro("interleave", &TensorMapEncodeTiledAttr::interleave, refl::DefaultValue(0))
      .def_ro("swizzle", &TensorMapEncodeTiledAttr::swizzle, refl::DefaultValue(0))
      .def_ro("l2_promotion", &TensorMapEncodeTiledAttr::l2_promotion, refl::DefaultValue(0))
      .def_ro("oob_fill", &TensorMapEncodeTiledAttr::oob_fill, refl::DefaultValue(0))
      .def_ro("force_cu_dtype", &TensorMapEncodeTiledAttr::force_cu_dtype, refl::DefaultValue(-1));
}
const Op& tensormap_encode_tiled_op() {
  static const Op op = Op::Get("tirx.cuda.tensormap_encode_tiled");
  return op;
}
TVM_FFI_STATIC_INIT_BLOCK() {
  TensorMapEncodeTiledAttr::RegisterReflection();
  ffi::reflection::GlobalDef().def(
      "tirx.cuda.TensorMapEncodeTiledAttr",
      [](DLDataType descriptor_dtype, int64_t rank, int64_t interleave, int64_t swizzle,
         int64_t l2_promotion, int64_t oob_fill, int64_t force_cu_dtype) {
        auto attrs = ffi::make_object<TensorMapEncodeTiledAttr>();
        attrs->descriptor_dtype = descriptor_dtype;
        attrs->rank = rank;
        attrs->interleave = interleave;
        attrs->swizzle = swizzle;
        attrs->l2_promotion = l2_promotion;
        attrs->oob_fill = oob_fill;
        attrs->force_cu_dtype = force_cu_dtype;
        return Attrs(attrs);
      });
  OpDef("tirx.cuda.tensormap_encode_tiled")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg("descriptor", "The descriptor."), sig::arg("data", "The input data."),
                 sig::var_args<PrimExpr>("args"), sig::call_attrs<TensorMapEncodeTiledAttr>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.cuda.tensormap_encode_tiled"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace cuda
}  // namespace backend
}  // namespace tvm
