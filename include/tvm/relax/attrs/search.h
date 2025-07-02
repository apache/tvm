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
 * \file tvm/relax/attrs/search.h
 * \brief Attributes for search operators.
 */
#ifndef TVM_RELAX_ATTRS_SEARCH_H_
#define TVM_RELAX_ATTRS_SEARCH_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes for search operators */
struct ArgmaxArgminAttrs : public AttrsNodeReflAdapter<ArgmaxArgminAttrs> {
  Optional<int64_t> axis;
  bool keepdims;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ArgmaxArgminAttrs>()
        .def_ro("axis", &ArgmaxArgminAttrs::axis,
                "The axis along which to perform the argmin/argmax.")
        .def_ro("keepdims", &ArgmaxArgminAttrs::keepdims,
                "If this is set to `True`, the reduced axis is left in the result as dimension "
                "with size "
                "one.");
  }

  static constexpr const char* _type_key = "relax.attrs.ArgmaxArgminAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(ArgmaxArgminAttrs, BaseAttrsNode);
};  // struct ArgmaxArgminAttrs

/*! \brief Attributes for bucketize operator */
struct BucketizeAttrs : public tvm::AttrsNodeReflAdapter<BucketizeAttrs> {
  bool out_int32;
  bool right;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BucketizeAttrs>()
        .def_ro("out_int32", &BucketizeAttrs::out_int32,
                "Indicate the output datatype, int32 if True, int64 otherwise.")
        .def_ro("right", &BucketizeAttrs::right,
                "Determines the behavior for values in boundaries");
  }

  static constexpr const char* _type_key = "relax.attrs.BucketizeAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(BucketizeAttrs, BaseAttrsNode);
};  // struct BucketizeAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_SEARCH_H_
