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
 *
 * \file src/relay/op/memory/on_device.cc
 * \brief Helpers for working with the "on_device" 'annotation' call.
 */

#include "./on_device.h"

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include "../../transforms/infer_layout_utils.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(OnDeviceAttrs);

const Op& OnDeviceOp() {
  static const Op& op = Op::Get("on_device");
  return op;
}

Call OnDevice(Expr body, VirtualDevice virtual_device, bool constrain_result, bool constrain_body) {
  ICHECK((!constrain_result && !constrain_body) || !virtual_device->IsFullyUnconstrained());
  auto attrs = make_object<OnDeviceAttrs>();
  attrs->virtual_device = (constrain_result || constrain_body)
                              ? std::move(virtual_device)
                              : VirtualDevice::FullyUnconstrained();
  attrs->constrain_result = constrain_result;
  attrs->constrain_body = constrain_body;
  Span span = body->span;  // about to be moved
  return Call(OnDeviceOp(), {std::move(body)}, Attrs(std::move(attrs)), /*type_args=*/{},
              std::move(span));
}

TVM_REGISTER_GLOBAL("relay.op.annotation._make.OnDevice").set_body_typed(OnDevice);

Expr MaybeOnDevice(Expr body, VirtualDevice virtual_device, bool constrain_result,
                   bool constrain_body) {
  if (virtual_device->IsFullyUnconstrained()) {
    // Nothing to annotate with.
    return body;
  }
  if (body->IsInstance<OpNode>() || body->IsInstance<ConstructorNode>()) {
    // These operators are device polymorphic so no annotation is required.
    return body;
  }
  if (body->IsInstance<GlobalVarNode>() || body->IsInstance<VarNode>()) {
    // The device can be recovered from the binding site of the global or local variable.
    return body;
  }
  if (body->IsInstance<FunctionNode>()) {
    // If a primitive function then it is device polymorphic. Otherwise the device is captured
    // by the function's "result_virtual_device" attribute.
    return body;
  }
  OnDeviceProps props = GetOnDeviceProps(body);
  if (props.body.defined()) {
    // The user is asking for
    //   on_device(on_device(body, virtual_device=inner), virtual_device=outer)
    //   ^         ^         ^
    //   outer     middle    inner
    // First recover the implied constraints (if any) for outer and inner, and check they don't
    // contradict.
    const VirtualDevice& inner = props.virtual_device;
    const VirtualDevice& outer = virtual_device;
    bool constrain_outer = constrain_result;
    bool constrain_inner = props.constrain_body;
    if (constrain_outer && constrain_inner) {
      ICHECK(inner == outer) << "Cannot constrain result and body of nested on_device calls to "
                                "different virtual devices";
    }
    // There are two possible ways the middle sub-expression may be constrained, check they don't
    // contradict.
    bool constrain_middle_via_outer = constrain_body;
    bool constrain_middle_via_inner = props.constrain_result;
    if (constrain_middle_via_outer && constrain_middle_via_inner) {
      ICHECK(inner == outer) << "Cannot constrain intermediate result of nested on_device calls to "
                                "different virtual devices";
    }
    // We can now ignore the middle constraint.
    // If the outer on_device has any constraint then use virtual_device given for it.
    // Otherwise we can use the existing inner virtual_device.
    return OnDevice(props.body, (constrain_inner || constrain_outer) ? outer : inner,
                    constrain_outer, constrain_inner);
  } else {
    return OnDevice(body, std::move(virtual_device), constrain_result, constrain_body);
  }
}

RELAY_REGISTER_OP("on_device")
    .describe(R"code(Annotate an expression with device type)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("body", "Expr", "The sub-expression to be annotated.")
    .set_support_level(10)
    .add_type_rel("Identity", IdentityRel)
    .set_attrs_type_key("relay.attrs.OnDeviceAttrs")
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<TNonComputational>("TNonComputational", true);

OnDeviceProps GetOnDeviceProps(const CallNode* call_node) {
  if (call_node->op == OnDeviceOp()) {
    ICHECK_EQ(call_node->args.size(), 1) << "on_device expects one argument";
    ICHECK(call_node->attrs.defined()) << "on_device requires attributes";
    const auto* on_device_attrs = call_node->attrs.as<OnDeviceAttrs>();
    ICHECK(on_device_attrs != nullptr) << "on_device requires OnDeviceAttrs";
    return {call_node->args[0], on_device_attrs->virtual_device, on_device_attrs->constrain_result,
            on_device_attrs->constrain_body};
  }
  return {};
}

OnDeviceProps GetOnDeviceProps(const Expr& expr) {
  if (const auto* call_node = expr.as<CallNode>()) {
    return GetOnDeviceProps(call_node);
  }
  return {};
}

}  // namespace relay
}  // namespace tvm
