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
struct SortAttrs : public tvm::AttrsNode<SortAttrs> {
  int axis;
  bool descending;

  TVM_DECLARE_ATTRS(SortAttrs, "relax.attrs.SortAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1).describe(
        "Axis along which the sort is computed."
        "The default the last axis is used.");
    TVM_ATTR_FIELD(descending)
        .set_default(false)
        .describe(
            "Whether to sort in descending order."
            "If it is not specified, it defaults to the ascending order.");
  }
};  // struct SortAttrs

/*! \brief Attributes used in argsort operator */
struct ArgsortAttrs : public tvm::AttrsNode<ArgsortAttrs> {
  int axis;
  bool descending;
  DataType dtype;

  TVM_DECLARE_ATTRS(ArgsortAttrs, "relax.attrs.ArgsortAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1).describe(
        "Axis along which the argsort is computed."
        "The default the last axis is used.");
    TVM_ATTR_FIELD(descending)
        .set_default(false)
        .describe(
            "Whether to argsort in descending order."
            "If it is not specified, it defaults to the ascending order.");
    TVM_ATTR_FIELD(dtype)
        .set_default(NullValue<DataType>())
        .describe("DType of the output indices.");
  }
};  // struct ArgsortAttrs

/*! \brief Attributes used in topk operator */
struct TopKAttrs : public tvm::AttrsNode<TopKAttrs> {
  int k;
  int axis;
  bool largest;
  String ret_type;
  DataType dtype;

  TVM_DECLARE_ATTRS(TopKAttrs, "relax.attrs.TopKAttrs") {
    TVM_ATTR_FIELD(k).describe("Number of top elements to select");
    TVM_ATTR_FIELD(axis).set_default(-1).describe("Axis along which to sort the input tensor.");
    TVM_ATTR_FIELD(ret_type).set_default("both").describe(
        "The return type [both, values, indices]."
        "both - return both top k data and indices."
        "values - return top k data only."
        "indices - return top k indices only.");
    TVM_ATTR_FIELD(largest).set_default(true).describe(
        "Whether to return largest or smallest elements."
        "By default, return the largest k elements.");
    TVM_ATTR_FIELD(dtype)
        .set_default(NullValue<DataType>())
        .describe("Data type of the output indices.");
  }
};  // struct TopKAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_SORTING_H_
