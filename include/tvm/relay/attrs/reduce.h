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
 * \file tvm/relay/attrs/reduce.h
 * \brief Auxiliary attributes for reduce operators.
 */
#ifndef TVM_RELAY_ATTRS_REDUCE_H_
#define TVM_RELAY_ATTRS_REDUCE_H_

#include <tvm/ir/attrs.h>

#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes for Reduce operators */
struct ReduceAttrs : public tvm::AttrsNode<ReduceAttrs> {
  Array<Integer> axis;
  bool keepdims;
  bool exclude;

  TVM_DECLARE_ATTRS(ReduceAttrs, "relay.attrs.ReduceAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Array<Integer>>())
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");

    TVM_ATTR_FIELD(keepdims).set_default(false).describe(
        "If this is set to `True`, the reduced axes are left "
        "in the result as dimension with size one.");
    TVM_ATTR_FIELD(exclude).set_default(false).describe(
        "Whether to perform reduction on axis that are NOT in axis instead.");
  }
};

/*! \brief Attributes for Reduce operators which reduce by finding a single element. E.g. argmin */
struct ArgReduceAttrs : public tvm::AttrsNode<ArgReduceAttrs> {
  Array<Integer> axis;
  bool keepdims;
  bool select_last_index;
  bool exclude;

  TVM_DECLARE_ATTRS(ArgReduceAttrs, "relay.attrs.ArgReduceAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Array<Integer>>())
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");

    TVM_ATTR_FIELD(keepdims).set_default(false).describe(
        "If this is set to `True`, the reduced axes are left "
        "in the result as dimension with size one.");
    TVM_ATTR_FIELD(select_last_index)
        .set_default(false)
        .describe(
            "Whether to select the last index if the target element appears multiple times, else "
            "select the first index which the target element appears");
    TVM_ATTR_FIELD(exclude).set_default(false).describe(
        "Whether to perform reduction on axis that are NOT in axis instead.");
  }
};

struct VarianceAttrs : public tvm::AttrsNode<VarianceAttrs> {
  Array<Integer> axis;
  bool keepdims;
  bool exclude;
  bool unbiased;

  TVM_DECLARE_ATTRS(VarianceAttrs, "relay.attrs.VarianceAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Array<Integer>>())
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");

    TVM_ATTR_FIELD(keepdims).set_default(false).describe(
        "If this is set to `True`, the reduced axes are left "
        "in the result as dimension with size one.");
    TVM_ATTR_FIELD(exclude).set_default(false).describe(
        "Whether to perform reduction on axis that are NOT in axis instead.");
    TVM_ATTR_FIELD(unbiased).set_default(false).describe("Whether to use the unbiased estimation.");
  }
};
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_REDUCE_H_
