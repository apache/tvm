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
 * \file tvm/relax/attrs/vision.h
 * \brief Auxiliary attributes for vision operators.
 */
#ifndef TVM_RELAX_ATTRS_VISION_H_
#define TVM_RELAX_ATTRS_VISION_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in AllClassNonMaximumSuppression operator */
struct AllClassNonMaximumSuppressionAttrs
    : public tvm::AttrsNode<AllClassNonMaximumSuppressionAttrs> {
  String output_format;

  TVM_DECLARE_ATTRS(AllClassNonMaximumSuppressionAttrs,
                    "relax.attrs.AllClassNonMaximumSuppressionAttrs") {
    TVM_ATTR_FIELD(output_format)
        .set_default("onnx")
        .describe(
            "Output format, onnx or tensorflow. Returns outputs in a way that can be easily "
            "consumed by each frontend.");
  }
};  // struct AllClassNonMaximumSuppressionAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_VISION_H_
