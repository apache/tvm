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
 * \file tvm/relax/attrs/sorting.h
 * \brief Attributes for sorting operators.
 */
#ifndef TVM_RELAX_ATTRS_SORTING_H_
#define TVM_RELAX_ATTRS_SORTING_H_

#include <tvm/relax/expr.h>
#include <tvm/tir/index_map.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in sort operator */
struct SortAttrs : public AttrsNodeReflAdapter<SortAttrs> {
  int axis;
  bool descending;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SortAttrs>()
        .def_ro("axis", &SortAttrs::axis,
                "Axis along which the sort is computed."
                "The default the last axis is used.",
                refl::DefaultValue(-1))
        .def_ro("descending", &SortAttrs::descending,
                "Whether to sort in descending order."
                "If it is not specified, it defaults to the ascending order.",
                refl::DefaultValue(false));
  }

  static constexpr const char* _type_key = "relax.attrs.SortAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(SortAttrs, BaseAttrsNode);
};  // struct SortAttrs

/*! \brief Attributes used in argsort operator */
struct ArgsortAttrs : public AttrsNodeReflAdapter<ArgsortAttrs> {
  int axis;
  bool descending;
  DataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ArgsortAttrs>()
        .def_ro("axis", &ArgsortAttrs::axis,
                "Axis along which the argsort is computed."
                "The default the last axis is used.",
                refl::DefaultValue(-1))
        .def_ro("descending", &ArgsortAttrs::descending,
                "Whether to argsort in descending order."
                "If it is not specified, it defaults to the ascending order.",
                refl::DefaultValue(false))
        .def_ro("dtype", &ArgsortAttrs::dtype, "DType of the output indices.",
                refl::DefaultValue(NullValue<DataType>()));
  }

  static constexpr const char* _type_key = "relax.attrs.ArgsortAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(ArgsortAttrs, BaseAttrsNode);
};  // struct ArgsortAttrs

/*! \brief Attributes used in topk operator */
struct TopKAttrs : public AttrsNodeReflAdapter<TopKAttrs> {
  int k;
  int axis;
  bool largest;
  String ret_type;
  DataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TopKAttrs>()
        .def_ro("k", &TopKAttrs::k, "Number of top elements to select")
        .def_ro("axis", &TopKAttrs::axis, "Axis along which to sort the input tensor.",
                refl::DefaultValue(-1))
        .def_ro("ret_type", &TopKAttrs::ret_type,
                "The return type [both, values, indices]."
                "both - return both top k data and indices."
                "values - return top k data only."
                "indices - return top k indices only.",
                refl::DefaultValue("both"))
        .def_ro("largest", &TopKAttrs::largest,
                "Whether to return largest or smallest elements."
                "By default, return the largest k elements.",
                refl::DefaultValue(true))
        .def_ro("dtype", &TopKAttrs::dtype, "Data type of the output indices.",
                refl::DefaultValue(NullValue<DataType>()));
  }

  static constexpr const char* _type_key = "relax.attrs.TopKAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(TopKAttrs, BaseAttrsNode);
};  // struct TopKAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_SORTING_H_
