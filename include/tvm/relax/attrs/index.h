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
 * \file tvm/relax/attrs/index.h
 * \brief Attributes for indexing operators.
 */
#ifndef TVM_RELAX_ATTRS_INDEX_H_
#define TVM_RELAX_ATTRS_INDEX_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in take operator */
struct TakeAttrs : public AttrsNodeReflAdapter<TakeAttrs> {
  Optional<int64_t> axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TakeAttrs>().def_ro("axis", &TakeAttrs::axis,
                                        "The axis over which to select values.");
  }

  static constexpr const char* _type_key = "relax.attrs.TakeAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(TakeAttrs, BaseAttrsNode);
};  // struct TakeAttrs

/*! \brief Attributes used in strided_slice operator */
struct StridedSliceAttrs : public AttrsNodeReflAdapter<StridedSliceAttrs> {
  bool assume_inbound;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StridedSliceAttrs>().def_ro(
        "assume_inbound", &StridedSliceAttrs::assume_inbound,
        "Whether to assume the indices are in bound. If it is set to false, "
        "out of bound indices will be clipped to the bound.",
        refl::DefaultValue(true));
  }

  static constexpr const char* _type_key = "relax.attrs.StridedSliceAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(StridedSliceAttrs, BaseAttrsNode);
};  // struct StridedSliceAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_INDEX_H_
