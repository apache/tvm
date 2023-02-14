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
 * \file tvm/relax/attrs/set.h
 * \brief Attributes for set operators.
 */
#ifndef TVM_RELAX_ATTRS_SET_H_
#define TVM_RELAX_ATTRS_SET_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in unique operator */
struct UniqueAttrs : public tvm::AttrsNode<UniqueAttrs> {
  bool sorted;
  bool return_index;
  bool return_inverse;
  bool return_counts;
  Optional<Integer> axis;

  TVM_DECLARE_ATTRS(UniqueAttrs, "relax.attrs.UniqueAttrs") {
    TVM_ATTR_FIELD(sorted).describe(
        "Whether to sort the unique elements in ascending order before returning as output.");
    TVM_ATTR_FIELD(return_index)
        .describe(
            "Whether to return an additional tensor with indices for where elements in the unique "
            "tensor come from the original input.");
    TVM_ATTR_FIELD(return_inverse)
        .describe(
            "Whether to return an additional tensor with indices for where elements in the "
            "original input ended up in the returned unique list.");
    TVM_ATTR_FIELD(return_counts)
        .describe("Whether to return an additional tensor with counts of each unique elements");
    TVM_ATTR_FIELD(axis).describe(
        "The dimension to apply unique. If it is NullOpt, the unique values of the flattened input "
        "is are returned.");
  }
};  // struct UniqueAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_SET_H_
