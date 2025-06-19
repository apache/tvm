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
 * \file tvm/relax/attrs/ccl.h
 * \brief Attributes for ccl operators.
 */
#ifndef TVM_RELAX_ATTRS_CCL_H_
#define TVM_RELAX_ATTRS_CCL_H_

#include <tvm/ffi/reflection/reflection.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in allreduce operators */
struct AllReduceAttrs : public tvm::AttrsNodeReflAdapter<AllReduceAttrs> {
  String op_type;
  bool in_group;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllReduceAttrs>()
        .def_ro("op_type", &AllReduceAttrs::op_type,
                "The type of reduction operation to be applied to the input data. Now only sum is "
                "supported.")
        .def_ro("in_group", &AllReduceAttrs::in_group,
                "Whether the reduction operation performs in group or globally or in group as "
                "default.");
  }

  static constexpr const char* _type_key = "relax.attrs.AllReduceAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(AllReduceAttrs, BaseAttrsNode);
};  // struct AllReduceAttrs

/*! \brief Attributes used in allgather operators */
struct AllGatherAttrs : public tvm::AttrsNodeReflAdapter<AllGatherAttrs> {
  int num_workers;
  bool in_group;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllGatherAttrs>()
        .def_ro("num_workers", &AllGatherAttrs::num_workers,
                "The number of workers, also the number of parts the given buffer should be "
                "chunked into.")
        .def_ro("in_group", &AllGatherAttrs::in_group,
                "Whether the allgather operation performs in group or globally or in group as "
                "default.");
  }

  static constexpr const char* _type_key = "relax.attrs.AllGatherAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(AllGatherAttrs, BaseAttrsNode);
};  // struct AllGatherAttrs

/*! \brief Attributes used in scatter operators */
struct ScatterCollectiveAttrs : public tvm::AttrsNodeReflAdapter<ScatterCollectiveAttrs> {
  int num_workers;
  int axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScatterCollectiveAttrs>()
        .def_ro("num_workers", &ScatterCollectiveAttrs::num_workers,
                "The number of workers, also the number of parts the given buffer should be "
                "chunked into.")
        .def_ro("axis", &ScatterCollectiveAttrs::axis,
                "The axis of the tensor to be scattered. The tensor will be chunked along "
                "this axis.");
  }

  static constexpr const char* _type_key = "relax.attrs.ScatterCollectiveAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(ScatterCollectiveAttrs, BaseAttrsNode);
};  // struct ScatterCollectiveAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_CCL_H_
