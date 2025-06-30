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
 * \file tvm/relax/attrs/datatype.h
 * \brief Attributes for datatype operators.
 */
#ifndef TVM_RELAX_ATTRS_DATATYPE_H_
#define TVM_RELAX_ATTRS_DATATYPE_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in astype operator */
struct AstypeAttrs : public AttrsNodeReflAdapter<AstypeAttrs> {
  DataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AstypeAttrs>().def_ro("dtype", &AstypeAttrs::dtype, "Target data type");
  }

  static constexpr const char* _type_key = "relax.attrs.AstypeAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(AstypeAttrs, BaseAttrsNode);
};  // struct AstypeAttrs.

/*! \brief Attributes used in wrap_param operator */
struct WrapParamAttrs : public AttrsNodeReflAdapter<WrapParamAttrs> {
  DataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WrapParamAttrs>().def_ro("dtype", &WrapParamAttrs::dtype, "Target data type");
  }

  static constexpr const char* _type_key = "relax.attrs.WrapParamAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(WrapParamAttrs, BaseAttrsNode);
};  // struct WrapParamAttrs.

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_DATATYPE_H_
