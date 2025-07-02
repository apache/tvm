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
 * \file tvm/relax/attrs/statistical.h
 * \brief Attributes for statistical operators.
 */
#ifndef TVM_RELAX_ATTRS_STATISTICAL_H_
#define TVM_RELAX_ATTRS_STATISTICAL_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes for statistical operators */
struct StatisticalAttrs : public AttrsNodeReflAdapter<StatisticalAttrs> {
  Optional<Array<Integer>> axis;
  bool keepdims;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StatisticalAttrs>()
        .def_ro("axis", &StatisticalAttrs::axis,
                "The axis or axes along which to perform the reduction.")
        .def_ro("keepdims", &StatisticalAttrs::keepdims,
                "If this is set to `True`, the reduced axes are left in the result as dimension "
                "with size "
                "one.");
  }

  static constexpr const char* _type_key = "relax.attrs.StatisticalAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(StatisticalAttrs, BaseAttrsNode);
};  // struct StatisticalAttrs

/*! \brief Attributes used in scan operators like cumsum, cumprod */
struct ScanopAttrs : public AttrsNodeReflAdapter<ScanopAttrs> {
  Optional<int64_t> axis;
  DataType dtype;
  Bool exclusive = Bool(false);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScanopAttrs>()
        .def_ro("axis", &ScanopAttrs::axis,
                "The axis along which to perform the scan computation."
                "The default (None) is to compute over the flattened array.")
        .def_ro("dtype", &ScanopAttrs::dtype,
                "The output data type."
                "If dtype is not specified, it defaults to the dtype of input data.")
        .def_ro("exclusive", &ScanopAttrs::exclusive, "The first element is not included",
                refl::DefaultValue(Bool(false)));
  }

  static constexpr const char* _type_key = "relax.attrs.ScanopAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(ScanopAttrs, BaseAttrsNode);
};  // struct ScanopAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_STATISTICAL_H_
