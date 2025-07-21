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
 * \file tvm/relax/attrs/sampling.h
 * \brief Attributes for sampling operators.
 */
#ifndef TVM_RELAX_ATTRS_SAMPLING_H_
#define TVM_RELAX_ATTRS_SAMPLING_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in multinomial_from_uniform operator */
struct MultinomialFromUniformAttrs : public AttrsNodeReflAdapter<MultinomialFromUniformAttrs> {
  DataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MultinomialFromUniformAttrs>().def_ro(
        "dtype", &MultinomialFromUniformAttrs::dtype, "Data type of the output indices.",
        refl::DefaultValue(DataType::Int(64)));
  }

  static constexpr const char* _type_key = "relax.attrs.MultinomialFromUniformAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(MultinomialFromUniformAttrs, BaseAttrsNode);
};  // struct MultinomialFromUniformAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_SAMPLING_H_
