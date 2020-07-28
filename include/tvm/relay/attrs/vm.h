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
 * \file tvm/relay/attrs/vm.h
 * \brief Attributes for Relay vm operators.
 */
#ifndef TVM_RELAY_ATTRS_VM_H_
#define TVM_RELAY_ATTRS_VM_H_

#include <tvm/ir/attrs.h>

namespace tvm {
namespace relay {

/*!
 * \brief Options for the shape function operator.
 */
struct ShapeFuncAttrs : public tvm::AttrsNode<ShapeFuncAttrs> {
  Array<Integer> is_input;

  TVM_DECLARE_ATTRS(ShapeFuncAttrs, "relay.attrs.ShapeFuncAttrs") {
    TVM_ATTR_FIELD(is_input).describe(
        "A bool indicating whether the shape function should"
        "expect shape or input in each position.");
  }
};

/*!
 * \brief Attributes for VM reshape_tensor operator.
 */
struct ReshapeTensorAttrs : public tvm::AttrsNode<ReshapeTensorAttrs> {
  Array<PrimExpr> newshape;

  TVM_DECLARE_ATTRS(ReshapeTensorAttrs, "relay.attrs.ReshapeTensorAttrs") {
    TVM_ATTR_FIELD(newshape).describe("The new shape of output tensor");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_VM_H_
