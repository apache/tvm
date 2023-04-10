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
 * \file tvm/relax/attrs/linear_algebra.h
 * \brief Attributes for linear algebra operators.
 */
#ifndef TVM_RELAX_ATTRS_LINEAR_ALGEBRA_H_
#define TVM_RELAX_ATTRS_LINEAR_ALGEBRA_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes for matmul operator */
struct MatmulAttrs : public tvm::AttrsNode<MatmulAttrs> {
  DataType out_dtype;

  TVM_DECLARE_ATTRS(MatmulAttrs, "relax.attrs.MatmulAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe("The data type of the output tensor");
  }
};  // struct MatmulAttrs

/*! \brief Attributes used in einsum operator */
struct EinsumAttrs : public tvm::AttrsNode<EinsumAttrs> {
  String subscripts;

  TVM_DECLARE_ATTRS(EinsumAttrs, "relax.attrs.EinsumAttrs") {
    TVM_ATTR_FIELD(subscripts).describe("The einsum expression string");
  }
};  // struct EinsumAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_LINEAR_ALGEBRA_H_
