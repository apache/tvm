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
struct StatisticalAttrs : public tvm::AttrsNode<StatisticalAttrs> {
  Optional<Array<Integer>> axis;
  bool keepdims;

  TVM_DECLARE_ATTRS(StatisticalAttrs, "relax.attrs.StatisticalAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis or axes along which to perform the reduction.");
    TVM_ATTR_FIELD(keepdims).describe(
        "If this is set to `True`, the reduced axes are left in the result as dimension with size "
        "one.");
  }
};  // struct StatisticalAttrs

/*! \brief Attributes used in scan operators like cumsum, cumprod */
struct ScanopAttrs : public tvm::AttrsNode<ScanopAttrs> {
  Optional<Integer> axis;
  DataType dtype;
  Bool exclusive = Bool(false);

  TVM_DECLARE_ATTRS(ScanopAttrs, "relax.attrs.ScanopAttrs") {
    TVM_ATTR_FIELD(axis).describe(
        "The axis along which to perform the scan computation."
        "The default (None) is to compute over the flattened array.");
    TVM_ATTR_FIELD(dtype).describe(
        "The output data type."
        "If dtype is not specified, it defaults to the dtype of input data.");
    TVM_ATTR_FIELD(exclusive)
        .describe("The first element is not included")
        .set_default(Bool(false));
  }
};  // struct ScanopAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_STATISTICAL_H_
