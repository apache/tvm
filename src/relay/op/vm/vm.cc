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
 * \file src/relay/op/vm/vm.cc
 * \brief Dialect operators for Relay VM.
 */

#include <topi/elemwise.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/data_type.h>

#include "../../transforms/infer_layout_util.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

RELAY_REGISTER_OP("vm.shape_of")
    .describe(R"code(Get the shape of an input tensor.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("tensor", "Tensor", "The input tensor")
    .add_type_rel("ShapeOf", ShapeOfRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

TVM_REGISTER_GLOBAL("relay.op.vm.shape_of").set_body_typed([](Expr expr) {
  auto attrs = make_object<ShapeOfAttrs>();
  attrs->dtype = DataType::Int(64);
  static const Op& op = Op::Get("vm.shape_of");
  return Call(op, {expr}, Attrs(attrs), {});
});

}  // namespace relay
}  // namespace tvm
