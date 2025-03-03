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
struct QuantizeAttrs : public tvm::AttrsNode<QuantizeAttrs> {
  DataType out_dtype;
  int axis;

  TVM_DECLARE_ATTRS(QuantizeAttrs, "relax.attrs.QuantizeAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe("Output data type.");
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization/dequantization. "
            "Default value is -1, which corresponds to the last axis.")
        .set_default(-1);
  }
};  // QuantizeAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_QDQ_H_
