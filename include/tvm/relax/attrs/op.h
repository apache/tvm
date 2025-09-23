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
 * \file tvm/relax/attrs/op.h
 * \brief Attributes for relax specific operators.
 */
#ifndef TVM_RELAX_ATTRS_OP_H_
#define TVM_RELAX_ATTRS_OP_H_

#include <tvm/ir/global_info.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in call_tir_with_grad */
struct CallTIRWithGradAttrs : public AttrsNodeReflAdapter<CallTIRWithGradAttrs> {
  ffi::String te_grad_name;
  ffi::Map<ffi::String, Any> te_grad_kwargs;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallTIRWithGradAttrs>()
        .def_ro(
            "te_grad_name", &CallTIRWithGradAttrs::te_grad_name,
            "The name of the te gradient function associated with this call_tir_with_grad node.")
        .def_ro("te_grad_kwargs", &CallTIRWithGradAttrs::te_grad_kwargs,
                "The keyword arguments passed to the te gradient function.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.CallTIRWithGradAttrs", CallTIRWithGradAttrs,
                                    BaseAttrsNode);
};  // struct CallTIRAttrs

/*! \brief Attributes used in call_tir_inplace */
struct CallTIRInplaceAttrs : public AttrsNodeReflAdapter<CallTIRInplaceAttrs> {
  /*!
   * \brief Indices that describe which input corresponds to which output.
   *
   * If the `i`th member has the value `k` >= 0, then that means that input `k` should be used to
   * store the `i`th output. If an element has the value -1, that means a new tensor should be
   * allocated for that output.
   */
  ffi::Array<Integer> inplace_indices;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallTIRInplaceAttrs>().def_ro("inplace_indices",
                                                  &CallTIRInplaceAttrs::inplace_indices);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.CallTIRInplaceAttrs", CallTIRInplaceAttrs,
                                    BaseAttrsNode);
};  // struct CallTIRInplaceAttrs

/*! \brief Attributes used in call_inplace_packed */
struct CallInplacePackedAttrs : public AttrsNodeReflAdapter<CallInplacePackedAttrs> {
  /*!
   * \brief Indices that describe which input corresponds to which output.
   *
   * If the `i`th member has the value `k` >= 0, then that means that input `k` should be used to
   * store the `i`th output. If an element has the value -1, that means the output will be newly
   * allocated.
   */
  ffi::Array<Integer> inplace_indices;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallInplacePackedAttrs>().def_ro("inplace_indices",
                                                     &CallInplacePackedAttrs::inplace_indices);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.CallInplacePackedAttrs", CallInplacePackedAttrs,
                                    BaseAttrsNode);
};  // struct CallInplacePackedAttrs

/*! \brief Attributes used in to_vdevice */
struct ToVDeviceAttrs : public AttrsNodeReflAdapter<ToVDeviceAttrs> {
  VDevice dst_vdevice;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ToVDeviceAttrs>().def_ro("dst_vdevice", &ToVDeviceAttrs::dst_vdevice,
                                             "The destination device where the data is copied to.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ToVDeviceAttrs", ToVDeviceAttrs, BaseAttrsNode);
};  // struct ToVDeviceAttrs

/*! \brief Attributes used in hint_on_device */
struct HintOnDeviceAttrs : public AttrsNodeReflAdapter<HintOnDeviceAttrs> {
  int32_t device_type;
  int32_t index;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<HintOnDeviceAttrs>()
        .def_ro("device_type", &HintOnDeviceAttrs::device_type,
                "The device type where the data is supposed to be executed.")
        .def_ro("index", &HintOnDeviceAttrs::index, "The device id.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.HintOnDeviceAttrs", HintOnDeviceAttrs,
                                    BaseAttrsNode);
};  // struct HintOnDeviceAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_OP_H_
