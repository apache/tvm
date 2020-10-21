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
 * \file src/relay/op/device_copy.cc
 * \brief Crossing device data copy operator.
 *
 * The pattern of this operator is registered as kOpaque. Hence, it could be
 * used as "barrier" to avoid fusing operators belonging to differen devices.
 */

#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/expr.h>
#include <tvm/topi/elemwise.h>

#include "../transforms/infer_layout_utils.h"
#include "type_relations.h"

namespace tvm {
namespace relay {

// relay.device_copy
TVM_REGISTER_NODE_TYPE(DeviceCopyAttrs);

TVM_REGISTER_GLOBAL("relay.op._make.device_copy")
    .set_body_typed([](Expr data, int src_dev_type, int dst_dev_type) {
      auto attrs = make_object<DeviceCopyAttrs>();
      attrs->src_dev_type = src_dev_type;
      attrs->dst_dev_type = dst_dev_type;
      static const Op& op = Op::Get("device_copy");
      return Call(op, {data}, Attrs(attrs), {});
    });

RELAY_REGISTER_OP("device_copy")
    .describe(R"code(
Copy data from one tensor to another. The source and destination might be
on different devices.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
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

}  // namespace relay
}  // namespace tvm
