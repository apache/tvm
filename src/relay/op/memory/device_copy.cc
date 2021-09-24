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
 * \file relay/op/memory/device_copy.cc
 * \brief Helpers for working with "device_copy" attributes.
 */

#include "./device_copy.h"

#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/topi/elemwise.h>

#include "../../transforms/infer_layout_utils.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

// relay.device_copy
TVM_REGISTER_NODE_TYPE(DeviceCopyAttrs);

const Op& DeviceCopyOp() {
  static const Op& op = Op::Get("device_copy");
  return op;
}

Expr DeviceCopy(Expr expr, DLDeviceType src_dev_type, DLDeviceType dst_dev_type) {
  auto attrs = make_object<DeviceCopyAttrs>();
  attrs->src_dev_type = src_dev_type;
  attrs->dst_dev_type = dst_dev_type;
  Span span = expr->span;
  return Call(DeviceCopyOp(), {std::move(expr)}, Attrs(attrs), /*type_args=*/{}, span);
}

Expr OptDeviceCopy(Expr expr, DLDeviceType src_dev_type, DLDeviceType dst_dev_type) {
  if (src_dev_type == dst_dev_type) {
    return expr;
  }
  ICHECK_NE(src_dev_type, kInvalidDeviceType);
  ICHECK_NE(dst_dev_type, kInvalidDeviceType);
  return DeviceCopy(expr, src_dev_type, dst_dev_type);
}

TVM_REGISTER_GLOBAL("relay.op._make.device_copy")
    .set_body_typed([](Expr expr, int src_dev_type, int dst_dev_type) {
      return DeviceCopy(expr, static_cast<DLDeviceType>(src_dev_type),
                        static_cast<DLDeviceType>(dst_dev_type));
    });

RELAY_REGISTER_OP("device_copy")
    .describe(R"code(
Copy data from one tensor to another. The source and destination might be
on different devices.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input data.")
    .set_support_level(10)
    .add_type_rel("Identity", IdentityRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

DeviceCopyProps GetDeviceCopyProps(const CallNode* call_node) {
  if (call_node->op == DeviceCopyOp()) {
    ICHECK_EQ(call_node->args.size(), 1) << "device_copy expects one argument";
    ICHECK(call_node->attrs.defined()) << "device_copy requires attributes";
    const auto* device_copy_attrs = call_node->attrs.as<DeviceCopyAttrs>();
    ICHECK(device_copy_attrs != nullptr) << "device_copy requires DeviceCopyAttrs";
    auto src_dev_type = static_cast<DLDeviceType>(device_copy_attrs->src_dev_type);
    auto dst_dev_type = static_cast<DLDeviceType>(device_copy_attrs->dst_dev_type);
    // Follow nesting:
    //   device_copy(device_copy(expr, src_dev_type=1, dst_dev_type=2),
    //               src_dev_type=2, dst_dev_type=3) ==> {expr, 1, 3}
    auto inner = GetDeviceCopyProps(call_node->args[0]);
    if (inner.body.defined()) {
      return {inner.body, inner.src_dev_type, inner.dst_dev_type};
    } else {
      return {call_node->args[0], src_dev_type, dst_dev_type};
    }
  }
  return {};
}

DeviceCopyProps GetDeviceCopyProps(const Expr& expr) {
  if (const auto* call_node = expr.as<CallNode>()) {
    return GetDeviceCopyProps(call_node);
  }
  return {};
}

}  // namespace relay
}  // namespace tvm
