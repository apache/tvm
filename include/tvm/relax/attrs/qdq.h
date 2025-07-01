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
 * \file include/tvm/relax/attrs/qdq.h
 * \brief Attributes for quantize/dequantize operators.
 */
#ifndef TVM_RELAX_ATTRS_QDQ_H_
#define TVM_RELAX_ATTRS_QDQ_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes for relax.quantize/relax.dequantize operator */
struct QuantizeAttrs : public AttrsNodeReflAdapter<QuantizeAttrs> {
  DataType out_dtype;
  int axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<QuantizeAttrs>()
        .def_ro("out_dtype", &QuantizeAttrs::out_dtype, "Output data type.")
        .def_ro("axis", &QuantizeAttrs::axis,
                "The output channel axis for channel wise quantization/dequantization. "
                "Default value is -1, which corresponds to the last axis.",
                refl::DefaultValue(-1));
  }

  static constexpr const char* _type_key = "relax.attrs.QuantizeAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(QuantizeAttrs, BaseAttrsNode);
};  // QuantizeAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_QDQ_H_
