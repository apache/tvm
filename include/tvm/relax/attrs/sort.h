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
 * \file tvm/relax/attrs/sort.h
 * \brief Attributes for sorting operators.
 */
#ifndef TVM_RELAX_ATTRS_SORT_H_
#define TVM_RELAX_ATTRS_SORT_H_

#include <tvm/relax/expr.h>
#include <tvm/tir/index_map.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in sort operator */
struct SortAttrs : public tvm::AttrsNode<SortAttrs> {
  Optional<Integer> axis;
  Optional<Bool> descending;

  TVM_DECLARE_ATTRS(SortAttrs, "relax.attrs.SortAttrs") {
    TVM_ATTR_FIELD(axis).describe(
        "Axis along which the sort is computed."
        "The default (None) is to compute the sort over the flattened array.");
    TVM_ATTR_FIELD(descending).describe(
        "Whether to sort in descending order."
        "If it is not specified, it defaults to the ascending order.");
  }
};  // struct SortAttrs
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_SORT_H_
