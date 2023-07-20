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

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in call_tir_with_grad */
struct CallTIRWithGradAttrs : public tvm::AttrsNode<CallTIRWithGradAttrs> {
  String te_grad_name;
  Map<String, ObjectRef> te_grad_kwargs;

  TVM_DECLARE_ATTRS(CallTIRWithGradAttrs, "relax.attrs.CallTIRWithGradAttrs") {
    TVM_ATTR_FIELD(te_grad_name)
        .describe(
            "The name of the te gradient function associated with this call_tir_with_grad node.");
    TVM_ATTR_FIELD(te_grad_kwargs)
        .describe("The keyword arguments passed to the te gradient function.");
  }
};  // struct CallTIRAttrs

/*! \brief Attributes used in call_tir_inplace */
struct CallTIRInplaceAttrs : public tvm::AttrsNode<CallTIRInplaceAttrs> {
  Array<Integer> inplace_indices;

  TVM_DECLARE_ATTRS(CallTIRInplaceAttrs, "relax.attrs.CallTIRInplaceAttrs") {
    TVM_ATTR_FIELD(inplace_indices)
        .describe(
            "Indices that describe which input corresponds to which output. If the `i`th member "
            "has the value `k` >= 0, then that means that input `k` should be used to store the "
            "`i`th output. If an element has the value -1, that means a new tensor should be "
            "allocated for that output.");
  }
};  // struct CallTIRInplaceAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_OP_H_
