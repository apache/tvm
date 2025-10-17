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

#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/type.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in AllClassNonMaximumSuppression operator */
struct AllClassNonMaximumSuppressionAttrs
    : public AttrsNodeReflAdapter<AllClassNonMaximumSuppressionAttrs> {
  ffi::String output_format;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllClassNonMaximumSuppressionAttrs>().def_ro(
        "output_format", &AllClassNonMaximumSuppressionAttrs::output_format,
        "Output format, onnx or tensorflow. Returns outputs in a way that can be easily "
        "consumed by each frontend.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AllClassNonMaximumSuppressionAttrs",
                                    AllClassNonMaximumSuppressionAttrs, BaseAttrsNode);
};  // struct AllClassNonMaximumSuppressionAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_VISION_H_
